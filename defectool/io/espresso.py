"""Quantum ESPRESSO input file generation and output parsing.

Template-based writer that gives full control over the input cards rather
than relying on ASE's Espresso interface (which makes too many assumptions
for HPC usage).
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from ..config import CalculatorConfig

logger = logging.getLogger(__name__)

# Bohr to Angstrom
BOHR_TO_ANG = 0.529177249
RY_TO_EV = 13.605693123


def _estimate_kmesh(cell: np.ndarray, kspacing: float) -> list[int]:
    """Estimate k-point mesh from cell vectors and target spacing (1/A)."""
    recip = 2 * np.pi * np.linalg.inv(cell).T
    recip_lengths = np.linalg.norm(recip, axis=1)
    mesh = [max(1, int(np.ceil(rl / kspacing))) for rl in recip_lengths]
    return mesh


def write_qe_input(
    atoms: Atoms,
    calc_config: CalculatorConfig,
    output_dir: Path,
    label: str = "relax",
    calculation: str = "relax",
) -> Path:
    """Write a complete QE pw.x input file.

    Parameters
    ----------
    atoms : ASE Atoms with cell and positions
    calc_config : calculator configuration block
    output_dir : directory to write into
    label : prefix for filenames
    calculation : 'relax', 'vc-relax', or 'scf'

    Returns
    -------
    Path to the written input file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pseudopotentials = calc_config.pseudopotentials or {}
    species = sorted(set(atoms.get_chemical_symbols()))

    # Validate pseudopotentials
    for sp in species:
        if sp not in pseudopotentials:
            raise ValueError(
                f"No pseudopotential specified for element '{sp}'. "
                f"Provide it in calculator.pseudopotentials."
            )

    cell = atoms.get_cell()
    kmesh = _estimate_kmesh(cell, calc_config.kspacing)
    nat = len(atoms)
    ntyp = len(species)

    lines = []
    lines.append("&CONTROL")
    lines.append(f"  calculation = '{calculation}'")
    lines.append(f"  prefix = '{label}'")
    lines.append(f"  outdir = './tmp'")
    lines.append(f"  pseudo_dir = './pseudo'")
    lines.append(f"  tprnfor = .true.")
    lines.append(f"  tstress = .true.")
    if calculation in ("relax", "vc-relax"):
        lines.append(f"  forc_conv_thr = 1.0d-4")
    lines.append("/")
    lines.append("")

    lines.append("&SYSTEM")
    lines.append(f"  ibrav = 0")
    lines.append(f"  nat = {nat}")
    lines.append(f"  ntyp = {ntyp}")
    lines.append(f"  ecutwfc = {calc_config.ecutwfc}")
    lines.append(f"  ecutrho = {calc_config.ecutrho}")
    lines.append(f"  occupations = 'smearing'")
    lines.append(f"  smearing = '{calc_config.smearing}'")
    lines.append(f"  degauss = {calc_config.degauss}")
    lines.append("/")
    lines.append("")

    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr = 1.0d-6")
    lines.append(f"  mixing_beta = 0.4")
    lines.append("/")
    lines.append("")

    if calculation in ("relax", "vc-relax"):
        lines.append("&IONS")
        lines.append(f"  ion_dynamics = 'bfgs'")
        lines.append("/")
        lines.append("")

    if calculation == "vc-relax":
        lines.append("&CELL")
        lines.append(f"  cell_dynamics = 'bfgs'")
        lines.append("/")
        lines.append("")

    # ATOMIC_SPECIES
    lines.append("ATOMIC_SPECIES")
    # We need atomic masses -- get from ASE
    from ase.data import atomic_masses, atomic_numbers
    for sp in species:
        mass = atomic_masses[atomic_numbers[sp]]
        pp = pseudopotentials[sp]
        lines.append(f"  {sp}  {mass:.4f}  {pp}")
    lines.append("")

    # CELL_PARAMETERS
    lines.append("CELL_PARAMETERS angstrom")
    for row in cell:
        lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
    lines.append("")

    # ATOMIC_POSITIONS
    lines.append("ATOMIC_POSITIONS angstrom")
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    for sym, pos in zip(symbols, positions):
        lines.append(f"  {sym}  {pos[0]:16.10f} {pos[1]:16.10f} {pos[2]:16.10f}")
    lines.append("")

    # K_POINTS
    lines.append(f"K_POINTS automatic")
    lines.append(f"  {kmesh[0]} {kmesh[1]} {kmesh[2]}  0 0 0")
    lines.append("")

    filepath = output_dir / f"{label}.pwi"
    filepath.write_text("\n".join(lines))
    logger.info("QE input written: %s", filepath)
    return filepath


def parse_qe_output(output_file: Path) -> Optional[dict]:
    """Parse a QE pw.x output file for energy and convergence info.

    Returns
    -------
    dict with keys: energy_eV, converged, n_steps, final_forces
    or None if parsing fails
    """
    output_file = Path(output_file)
    if not output_file.exists():
        logger.warning("QE output not found: %s", output_file)
        return None

    text = output_file.read_text()

    result = {
        "energy_eV": None,
        "converged": False,
        "n_steps": 0,
        "final_forces": [],
    }

    # Total energy (last occurrence)
    energy_matches = re.findall(
        r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text
    )
    if energy_matches:
        result["energy_eV"] = float(energy_matches[-1]) * RY_TO_EV

    # Convergence
    if "convergence has been achieved" in text.lower():
        result["converged"] = True

    # Number of BFGS steps
    step_matches = re.findall(r"number of bfgs steps\s+=\s+(\d+)", text)
    if step_matches:
        result["n_steps"] = int(step_matches[-1])

    # Final forces (last block)
    force_blocks = re.findall(
        r"Forces acting on atoms.*?\n((?:\s+atom\s+\d+.*\n)+)", text
    )
    if force_blocks:
        forces = []
        for line in force_blocks[-1].strip().split("\n"):
            m = re.search(
                r"force\s+=\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line
            )
            if m:
                forces.append([float(m.group(i)) for i in (1, 2, 3)])
        # QE forces are in Ry/Bohr, convert to eV/A
        result["final_forces"] = (
            np.array(forces) * RY_TO_EV / BOHR_TO_ANG
        ).tolist() if forces else []

    return result


def write_qe_batch(
    db,
    calc_config: CalculatorConfig,
    base_dir: Path,
    calculation: str = "relax",
) -> list[Path]:
    """Write QE input files for all generated structures in the database.

    Creates a directory tree: base_dir/<label>/relax.pwi
    """
    from ..db import STATUS_GENERATED

    base_dir = Path(base_dir)
    written = []

    rows = db.get_by_status(STATUS_GENERATED)
    for row_id, atoms, kvp in rows:
        label = kvp.get("label", f"id_{row_id}")
        safe_label = label.replace("+", "p").replace("-", "m").replace(".", "_")
        calc_dir = base_dir / safe_label
        path = write_qe_input(
            atoms, calc_config, calc_dir,
            label=safe_label, calculation=calculation,
        )
        written.append(path)

    # Pristine
    pristine_row = db.get_pristine()
    if pristine_row:
        _, pristine, _ = pristine_row
        calc_dir = base_dir / "pristine"
        path = write_qe_input(
            pristine, calc_config, calc_dir,
            label="pristine", calculation=calculation,
        )
        written.append(path)

    logger.info("Wrote %d QE input files under %s/", len(written), base_dir)
    return written
