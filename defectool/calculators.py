"""Calculator factory: all ASE-based calculators."""

import logging

from ase.calculators.calculator import Calculator

from .config import CalculatorConfig

logger = logging.getLogger(__name__)


def build_calculator(calc_config: CalculatorConfig) -> Calculator:
    """Build an ASE calculator from configuration."""
    name = calc_config.name

    if name == "mace_mp":
        return _build_mace_mp(calc_config)
    elif name == "mace":
        return _build_mace_custom(calc_config)
    elif name == "lammps":
        return _build_lammps(calc_config)
    elif name == "espresso":
        return _build_espresso(calc_config)
    else:
        raise ValueError(f"Unknown calculator: {name}")


def _build_mace_mp(c: CalculatorConfig) -> Calculator:
    try:
        from mace.calculators import mace_mp
    except ImportError:
        raise ImportError(
            "mace-torch required. Install: pip install mace-torch"
        )
    calc = mace_mp(model=c.model, device=c.device, default_dtype="float64")
    logger.info("MACE-MP ready (model=%s, device=%s)", c.model, c.device)
    return calc


def _build_mace_custom(c: CalculatorConfig) -> Calculator:
    try:
        from mace.calculators import MACECalculator
    except ImportError:
        raise ImportError(
            "mace-torch required. Install: pip install mace-torch"
        )
    if not c.model:
        raise ValueError("Custom MACE needs 'model' path.")
    calc = MACECalculator(
        model_paths=c.model, device=c.device, default_dtype="float64",
    )
    logger.info("MACE ready (model=%s)", c.model)
    return calc


def _build_lammps(c: CalculatorConfig) -> Calculator:
    try:
        from ase.calculators.lammpslib import LAMMPSlib
    except ImportError:
        raise ImportError("ASE LAMMPSlib not available.")
    cmds = [f"pair_style {c.pair_style}"]
    for coeff in c.pair_coeff:
        cmds.append(f"pair_coeff {coeff}")
    calc = LAMMPSlib(lmpcmds=cmds, log_file="lammps.log", keep_alive=True)
    logger.info("LAMMPS ready (pair_style=%s)", c.pair_style)
    return calc


def _build_espresso(c: CalculatorConfig) -> Calculator:
    from ase.calculators.espresso import Espresso
    input_data = {
        "system": {
            "ecutwfc": c.ecutwfc, "ecutrho": c.ecutrho,
            "occupations": "smearing", "smearing": c.smearing,
            "degauss": c.degauss,
        },
        "electrons": {"conv_thr": 1.0e-6, "mixing_beta": 0.4},
    }
    kwargs = {
        "pseudopotentials": c.pseudopotentials,
        "input_data": input_data,
        "kspacing": c.kspacing,
    }
    if c.pseudo_dir:
        kwargs["pseudo_dir"] = c.pseudo_dir
    calc = Espresso(**kwargs)
    logger.info("QE ready (ecutwfc=%.1f)", c.ecutwfc)
    return calc
