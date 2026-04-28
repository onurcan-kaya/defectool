"""Relaxation: drive ASE optimisers, export results as extended XYZ."""

import logging
import traceback
from pathlib import Path

from ase.io import write as ase_write
from ase.optimize import BFGS, FIRE, LBFGS

from .calculators import build_calculator
from .config import Config
from .db import DefectDB, STATUS_GENERATED

logger = logging.getLogger(__name__)

_OPTIMIZERS = {"BFGS": BFGS, "FIRE": FIRE, "LBFGS": LBFGS}


def relax_one(row_id, atoms, calculator, config, db, kvp) -> bool:
    """Relax a single structure. Returns True on success."""
    label = kvp.get("label", f"id_{row_id}")
    logger.info("Relaxing %s (id=%d, %d atoms)", label, row_id, len(atoms))

    db.set_status(row_id, "running")
    atoms.calc = calculator

    opt_name = config.relaxation.optimizer.upper()
    opt_cls = _OPTIMIZERS.get(opt_name)
    if opt_cls is None:
        raise ValueError(f"Unknown optimizer '{opt_name}'.")

    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    logfile = outdir / f"relax_{label}.log"

    try:
        # Get reference energy at step 0 for sanity checking
        initial_energy = atoms.get_potential_energy()

        opt = opt_cls(atoms, logfile=str(logfile))

        # Step-by-step relaxation with energy sanity check
        max_steps = config.relaxation.max_steps
        fmax_target = config.relaxation.fmax
        converged = False
        diverged = False

        for step in range(max_steps):
            # Check convergence
            forces = atoms.get_forces()
            fmax_now = abs(forces).max()
            if fmax_now < fmax_target:
                converged = True
                break

            # Check for energy divergence: if energy drops to more than
            # 10x the initial magnitude below the starting point, the
            # model is extrapolating into nonsense
            current_energy = atoms.get_potential_energy()
            threshold = 10.0 * max(abs(initial_energy), 10.0)
            if abs(current_energy - initial_energy) > threshold:
                logger.warning(
                    "  %s: energy diverged at step %d "
                    "(E=%.1f eV, started at %.1f eV). Aborting.",
                    label, step, current_energy, initial_energy,
                )
                diverged = True
                break

            opt.step()

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        final_fmax = float(abs(forces).max())
        n_steps = opt.nsteps

        if diverged:
            db.mark_failed(row_id, f"Energy diverged: {energy:.1f} eV")
            return False

        # Store in database
        db.store_relaxed(row_id, atoms, energy, n_steps,
                         final_fmax=final_fmax, converged=converged)

        if not converged:
            logger.warning(
                "  %s: max_steps reached. E=%.6f eV, fmax=%.4f eV/A",
                label, energy, final_fmax,
            )
        else:
            logger.info(
                "  %s: converged in %d steps. E=%.6f eV",
                label, n_steps, energy,
            )
        return True

    except Exception as e:
        logger.error("  %s FAILED: %s", label, e)
        logger.debug(traceback.format_exc())
        db.mark_failed(row_id, str(e))
        return False


def run_relaxation(config: Config, db: DefectDB) -> None:
    """Relax all unrelaxed structures."""
    logger.info("=== Relaxation ===")

    work = db.get_by_status(STATUS_GENERATED)
    if config.relaxation.restart:
        work += db.get_by_status("failed")

    if not work:
        logger.info("Nothing to relax.")
        return

    nprocs = max(1, config.nprocs)
    logger.info("%d structures to relax (%d workers).", len(work), nprocs)

    if nprocs == 1:
        _relax_serial(work, config, db)
    else:
        _relax_parallel(work, config, db, nprocs)


def _relax_serial(work, config, db):
    calculator = build_calculator(config.calculator)
    n_ok, n_fail = 0, 0
    for row_id, atoms, kvp in work:
        if relax_one(row_id, atoms, calculator, config, db, kvp):
            n_ok += 1
        else:
            n_fail += 1
    logger.info("Relaxation done: %d converged, %d failed.", n_ok, n_fail)


def _worker_init(calc_dict):
    """Initializer for each worker process. Builds the calculator once
    and stores it as a module-level global so it persists across tasks."""
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    global _worker_calc
    from .calculators import build_calculator as _build
    from .config import CalculatorConfig as _CC
    _worker_calc = _build(_CC(**calc_dict))


def _worker_relax(task):
    """Worker function. Uses the calculator built in _worker_init."""
    _logger = logging.getLogger(__name__)

    row_id, cell, positions, numbers, pbc, kvp, relax_dict, output_dir = task
    label = kvp.get("label", f"id_{row_id}")

    try:
        from ase import Atoms as _Atoms
        atoms = _Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
        atoms.calc = _worker_calc

        opt_name = relax_dict["optimizer"].upper()
        opt_cls = _OPTIMIZERS.get(opt_name)
        if opt_cls is None:
            return (row_id, False, None, None, 0, f"Unknown optimizer {opt_name}")

        from pathlib import Path as _Path
        _Path(output_dir).mkdir(parents=True, exist_ok=True)
        logfile = str(_Path(output_dir) / f"relax_{label}.log")

        opt = opt_cls(atoms, logfile=logfile)

        # Step-by-step with divergence check
        initial_energy = atoms.get_potential_energy()
        fmax_target = relax_dict["fmax"]
        max_steps = relax_dict["max_steps"]
        converged = False
        diverged = False

        for step in range(max_steps):
            forces = atoms.get_forces()
            if abs(forces).max() < fmax_target:
                converged = True
                break
            current_energy = atoms.get_potential_energy()
            threshold = 10.0 * max(abs(initial_energy), 10.0)
            if abs(current_energy - initial_energy) > threshold:
                _logger.warning(
                    "  %s: energy diverged at step %d (E=%.1f eV). Aborting.",
                    label, step, current_energy,
                )
                diverged = True
                break
            opt.step()

        if diverged:
            return (row_id, False, None, None, 0, 0.0, False,
                    f"Energy diverged: {atoms.get_potential_energy():.1f} eV")

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        final_fmax = float(abs(forces).max())
        n_steps = opt.nsteps

        result_atoms = {
            "cell": atoms.get_cell().tolist(),
            "positions": atoms.get_positions().tolist(),
            "numbers": atoms.get_atomic_numbers().tolist(),
            "pbc": atoms.get_pbc().tolist(),
        }

        if not converged:
            _logger.warning("  %s: max_steps reached. E=%.6f eV, fmax=%.4f", label, energy, final_fmax)
        else:
            _logger.info("  %s: converged in %d steps. E=%.6f eV", label, n_steps, energy)

        return (row_id, True, result_atoms, energy, n_steps, final_fmax, converged, "")

    except Exception as e:
        _logger.error("  %s FAILED: %s", label, e)
        return (row_id, False, None, None, 0, 0.0, False, str(e))


def _relax_parallel(work, config, db, nprocs):
    from concurrent.futures import ProcessPoolExecutor
    from ase import Atoms as _Atoms

    calc_dict = {
        "name": config.calculator.name,
        "model": config.calculator.model,
        "device": config.calculator.device,
        "pair_style": config.calculator.pair_style,
        "pair_coeff": config.calculator.pair_coeff,
        "pseudopotentials": config.calculator.pseudopotentials,
        "pseudo_dir": config.calculator.pseudo_dir,
        "ecutwfc": config.calculator.ecutwfc,
        "ecutrho": config.calculator.ecutrho,
        "kspacing": config.calculator.kspacing,
        "smearing": config.calculator.smearing,
        "degauss": config.calculator.degauss,
    }
    relax_dict = {
        "fmax": config.relaxation.fmax,
        "optimizer": config.relaxation.optimizer,
        "max_steps": config.relaxation.max_steps,
    }

    tasks = []
    for row_id, atoms, kvp in work:
        db.set_status(row_id, "running")
        task = (
            row_id,
            atoms.get_cell().tolist(),
            atoms.get_positions().tolist(),
            atoms.get_atomic_numbers().tolist(),
            atoms.get_pbc().tolist(),
            dict(kvp),
            relax_dict,
            config.output_dir,
        )
        tasks.append(task)

    n_ok, n_fail = 0, 0
    with ProcessPoolExecutor(
        max_workers=nprocs,
        initializer=_worker_init,
        initargs=(calc_dict,),
    ) as pool:
        for result in pool.map(_worker_relax, tasks):
            row_id, success, result_atoms, energy, n_steps, final_fmax, conv, reason = result
            if success and result_atoms is not None:
                atoms = _Atoms(
                    numbers=result_atoms["numbers"],
                    positions=result_atoms["positions"],
                    cell=result_atoms["cell"],
                    pbc=result_atoms["pbc"],
                )
                db.store_relaxed(row_id, atoms, energy, n_steps,
                                 final_fmax=final_fmax, converged=conv)
                n_ok += 1
            else:
                db.mark_failed(row_id, reason)
                n_fail += 1

    logger.info("Relaxation done: %d converged, %d failed.", n_ok, n_fail)


def export_xyz(config: Config, db: DefectDB) -> Path:
    """Export all converged structures to a single extended XYZ file."""
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    xyz_path = outdir / "relaxed_structures.xyz"

    rows = db.get_converged()
    if not rows:
        logger.warning("No converged structures to export.")
        return xyz_path

    all_atoms = []
    for row_id, atoms, kvp in rows:
        energy = kvp.get("total_energy")
        label = kvp.get("label", f"id_{row_id}")
        dtype = kvp.get("defect_type", "")
        dist_type = kvp.get("distortion_type", "")
        dist_mag = kvp.get("distortion_mag", 0.0)
        defect_config = kvp.get("defect_config", "default")

        # Strip calculator to avoid conflict with info keys
        atoms.calc = None
        atoms.info["REF_energy"] = energy
        atoms.info["label"] = label
        atoms.info["defect_type"] = dtype
        atoms.info["distortion_type"] = dist_type
        atoms.info["distortion_mag"] = dist_mag
        atoms.info["defect_config"] = defect_config
        atoms.info["config_type"] = f"{dtype}_{dist_type}_{dist_mag:+.3f}"

        all_atoms.append(atoms)

    ase_write(str(xyz_path), all_atoms, format="extxyz")
    logger.info("Exported %d structures to %s", len(all_atoms), xyz_path)
    return xyz_path
