"""LAMMPS data file and input script generation, plus log parsing."""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from ..config import CalculatorConfig

logger = logging.getLogger(__name__)


def write_lammps_data(atoms: Atoms, filepath: Path) -> Path:
    """Write a LAMMPS data file in atomic style."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    type_map = {s: i + 1 for i, s in enumerate(species)}

    cell = atoms.get_cell()
    positions = atoms.get_positions()

    # Triclinic box parameters
    a, b, c = cell[0], cell[1], cell[2]
    xlo, xhi = 0.0, np.linalg.norm(a)
    ylo, yhi = 0.0, np.linalg.norm(b)
    zlo, zhi = 0.0, np.linalg.norm(c)

    # Tilt factors
    xy = np.dot(b, a / np.linalg.norm(a))
    xz = np.dot(c, a / np.linalg.norm(a))
    yz_vec = b - xy * a / np.linalg.norm(a)
    yz = np.dot(c, yz_vec / np.linalg.norm(yz_vec)) if np.linalg.norm(yz_vec) > 1e-10 else 0.0

    lines = []
    lines.append("# LAMMPS data file written by defectool")
    lines.append("")
    lines.append(f"{len(atoms)} atoms")
    lines.append(f"{len(species)} atom types")
    lines.append("")
    lines.append(f"{xlo:.10f} {xhi:.10f} xlo xhi")
    lines.append(f"{ylo:.10f} {yhi:.10f} ylo yhi")
    lines.append(f"{zlo:.10f} {zhi:.10f} zlo zhi")

    if abs(xy) > 1e-10 or abs(xz) > 1e-10 or abs(yz) > 1e-10:
        lines.append(f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz")

    lines.append("")
    lines.append("Masses")
    lines.append("")
    for sp in species:
        mass = atomic_masses[atomic_numbers[sp]]
        lines.append(f"  {type_map[sp]}  {mass:.6f}  # {sp}")

    lines.append("")
    lines.append("Atoms")
    lines.append("")

    # Convert to fractional then to LAMMPS box coordinates
    frac = atoms.get_scaled_positions()
    for i, (sym, f_pos) in enumerate(zip(symbols, frac)):
        # LAMMPS Cartesian from fractional for triclinic box
        x = f_pos[0] * (xhi - xlo) + f_pos[1] * xy + f_pos[2] * xz
        y = f_pos[1] * (yhi - ylo) + f_pos[2] * yz
        z = f_pos[2] * (zhi - zlo)
        lines.append(
            f"  {i + 1}  {type_map[sym]}  {x:.10f}  {y:.10f}  {z:.10f}"
        )

    lines.append("")
    filepath.write_text("\n".join(lines))
    logger.info("LAMMPS data file written: %s", filepath)
    return filepath


def write_lammps_input(
    calc_config: CalculatorConfig,
    data_file: str = "structure.data",
    output_path: Path = Path("in.relax"),
    species: list[str] = None,
) -> Path:
    """Write a LAMMPS minimisation input script."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# LAMMPS relaxation script written by defectool")
    lines.append("units metal")
    lines.append("boundary p p p")
    lines.append("atom_style atomic")
    lines.append("")
    lines.append(f"read_data {data_file}")
    lines.append("")
    lines.append(f"pair_style {calc_config.pair_style}")
    for coeff in (calc_config.pair_coeff or []):
        lines.append(f"pair_coeff {coeff}")
    lines.append("")
    lines.append("thermo 10")
    lines.append("thermo_style custom step pe fnorm")
    lines.append("")
    lines.append("min_style cg")
    lines.append("minimize 1.0e-8 1.0e-4 10000 100000")
    lines.append("")
    lines.append("write_data relaxed.data")
    lines.append("print 'Final energy: $(pe) eV'")
    lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info("LAMMPS input script written: %s", output_path)
    return output_path


def write_lammps_batch(
    db,
    calc_config: CalculatorConfig,
    base_dir: Path,
) -> list[Path]:
    """Write LAMMPS data files and input scripts for all generated structures."""
    from ..db import STATUS_GENERATED

    base_dir = Path(base_dir)
    written = []

    species_set = set()

    rows = db.get_by_status(STATUS_GENERATED)
    for row_id, atoms, kvp in rows:
        label = kvp.get("label", f"id_{row_id}")
        safe_label = label.replace("+", "p").replace("-", "m").replace(".", "_")
        calc_dir = base_dir / safe_label
        species_set.update(atoms.get_chemical_symbols())

        data_path = write_lammps_data(atoms, calc_dir / "structure.data")
        input_path = write_lammps_input(
            calc_config,
            data_file="structure.data",
            output_path=calc_dir / "in.relax",
            species=sorted(species_set),
        )
        written.append(data_path)

    # Pristine
    pristine_row = db.get_pristine()
    if pristine_row:
        _, pristine, _ = pristine_row
        calc_dir = base_dir / "pristine"
        write_lammps_data(pristine, calc_dir / "structure.data")
        write_lammps_input(
            calc_config,
            data_file="structure.data",
            output_path=calc_dir / "in.relax",
        )

    logger.info("Wrote LAMMPS files for %d structures under %s/",
                len(written), base_dir)
    return written


def parse_lammps_log(log_file: Path) -> Optional[dict]:
    """Parse a LAMMPS log file for final energy."""
    log_file = Path(log_file)
    if not log_file.exists():
        return None

    text = log_file.read_text()

    result = {
        "energy_eV": None,
        "converged": False,
    }

    # Look for "Final energy:" line from our script
    m = re.search(r"Final energy:\s+([-\d.eE+]+)\s+eV", text)
    if m:
        result["energy_eV"] = float(m.group(1))

    # Check for minimisation convergence
    if "Minimization converged" in text:
        result["converged"] = True

    return result
