"""Structure generation: supercell, multiple defects, distortions."""

import logging

import numpy as np
import spglib
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

from .config import Config, DefectSpec
from .db import DefectDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supercell
# ---------------------------------------------------------------------------

def _auto_supercell(atoms: Atoms, min_length: float) -> list[int]:
    lengths = atoms.cell.lengths()
    reps = [max(1, int(np.ceil(min_length / l))) for l in lengths]
    logger.info(
        "Auto supercell: lengths [%.2f, %.2f, %.2f] -> %s",
        *lengths, reps,
    )
    return reps


def build_supercell(atoms: Atoms, config: Config) -> Atoms:
    if config.supercell is not None:
        reps = config.supercell
    else:
        reps = _auto_supercell(atoms, config.min_cell_length)
    supercell = atoms.repeat(reps)
    logger.info("Supercell: %d atoms, repetitions %s", len(supercell), reps)
    return supercell


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def find_inequivalent_sites(
    atoms: Atoms, element: str, symprec: float = 0.1,
) -> list[list[int]]:
    """Group sites of a given element by symmetry equivalence."""
    symbols = atoms.get_chemical_symbols()
    candidates = [i for i, s in enumerate(symbols) if s == element]
    if not candidates:
        raise ValueError(f"Element '{element}' not found.")

    cell = (
        atoms.get_cell(), atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    if dataset is None:
        logger.warning("spglib failed. Treating all sites as inequivalent.")
        return [[s] for s in candidates]

    equiv = dataset.equivalent_atoms
    groups: dict[int, list[int]] = {}
    for site in candidates:
        key = int(equiv[site])
        groups.setdefault(key, []).append(site)

    result = list(groups.values())
    logger.info(
        "%d %s sites -> %d inequivalent group(s)",
        len(candidates), element, len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Defect insertion (multiple defects)
# ---------------------------------------------------------------------------

def _pick_sites_for_defect(
    atoms: Atoms,
    spec: DefectSpec,
    rng: np.random.Generator,
    already_used: set[int],
) -> list[int]:
    """Pick atom indices for one DefectSpec, avoiding already_used sites."""
    if spec.site is not None:
        # Explicit site index
        return [spec.site]

    symbols = atoms.get_chemical_symbols()
    available = [
        i for i, s in enumerate(symbols)
        if s == spec.element and i not in already_used
    ]
    if len(available) < spec.count:
        raise ValueError(
            f"Need {spec.count} {spec.element} sites but only "
            f"{len(available)} available."
        )

    if spec.count == 1:
        # Use symmetry to pick representative site
        groups = find_inequivalent_sites(atoms, spec.element)
        # Filter out already used
        for group in groups:
            free = [s for s in group if s not in already_used]
            if free:
                return [free[0]]
        return [available[0]]
    else:
        # For multiple defects of same type, pick randomly
        chosen = rng.choice(available, size=spec.count, replace=False).tolist()
        return chosen


def insert_defects(
    atoms: Atoms,
    defect_specs: list[DefectSpec],
    seed: int = 42,
) -> tuple[Atoms, list[int], str]:
    """Apply all defects to a copy of the structure.

    Returns:
        defected: Atoms with all defects applied
        defect_sites: original indices of defect sites (before deletion)
        description: human-readable string of what was done
    """
    rng = np.random.default_rng(seed)
    defected = atoms.copy()
    used_sites: set[int] = set()
    all_defect_sites: list[int] = []
    descriptions: list[str] = []

    # Collect all operations first (vacancies must be applied last
    # because they change indices)
    vacancy_sites: list[int] = []
    substitutions: list[tuple[int, str]] = []
    antisites: list[int] = []

    for spec in defect_specs:
        sites = _pick_sites_for_defect(atoms, spec, rng, used_sites)
        used_sites.update(sites)
        all_defect_sites.extend(sites)

        if spec.dtype == "vacancy":
            vacancy_sites.extend(sites)
            descriptions.append(f"V_{spec.element}x{len(sites)}")
        elif spec.dtype == "substitution":
            for s in sites:
                substitutions.append((s, spec.substitute))
            descriptions.append(
                f"{spec.element}->{spec.substitute}x{len(sites)}"
            )
        elif spec.dtype == "antisite":
            antisites.extend(sites)
            descriptions.append(f"AS_{spec.element}x{len(sites)}")

    # Apply substitutions
    for site_idx, new_symbol in substitutions:
        old = defected[site_idx].symbol
        defected[site_idx].symbol = new_symbol
        logger.debug("Substitution: site %d %s -> %s", site_idx, old, new_symbol)

    # Apply antisites
    for site_idx in antisites:
        symbols = defected.get_chemical_symbols()
        original = symbols[site_idx]
        i_arr, j_arr, d_arr = neighbor_list("ijd", defected, cutoff=4.0)
        mask = i_arr == site_idx
        neighbours = j_arr[mask]
        distances = d_arr[mask]
        order = np.argsort(distances)
        swapped = False
        for idx in order:
            j = neighbours[idx]
            if symbols[j] != original and j not in used_sites:
                defected[site_idx].symbol = symbols[j]
                defected[j].symbol = original
                used_sites.add(j)
                logger.debug(
                    "Antisite: %d (%s) <-> %d (%s)",
                    site_idx, original, j, symbols[j],
                )
                swapped = True
                break
        if not swapped:
            raise ValueError(
                f"No valid antisite partner for site {site_idx}."
            )

    # Apply vacancies (delete in reverse order to preserve indices)
    for site_idx in sorted(vacancy_sites, reverse=True):
        del defected[site_idx]
        logger.debug("Vacancy: removed site %d", site_idx)

    description = "_".join(descriptions)
    logger.info("Defects applied: %s", description)
    return defected, all_defect_sites, description


# ---------------------------------------------------------------------------
# Distortions
# ---------------------------------------------------------------------------

def _get_neighbours(atoms, site, cutoff=3.5):
    i_arr, j_arr, d_arr = neighbor_list("ijd", atoms, cutoff=cutoff)
    mask = i_arr == site
    return j_arr[mask], d_arr[mask]


def generate_bond_distortion(
    atoms: Atoms,
    defect_sites: list[int],
    vacancy_sites: list[int],
    pristine: Atoms,
    original_defect_sites: list[int],
    fraction: float,
    cutoff: float = 3.5,
) -> Atoms:
    """Distort bonds around all defect sites by a fractional amount."""
    distorted = atoms.copy()
    positions = distorted.get_positions()

    for site in defect_sites:
        if site in vacancy_sites:
            # For vacancies: use pristine reference to find neighbours
            pristine_pos = pristine.get_positions()
            vacancy_pos = pristine_pos[site]
            n_idx, _ = _get_neighbours(pristine, site, cutoff)
            for j in n_idx:
                # Map pristine index to defected index (shifted by deletions)
                j_def = j
                for vs in sorted(vacancy_sites):
                    if vs < j:
                        j_def -= 1
                    elif vs == j:
                        j_def = -1
                        break
                if j_def < 0 or j_def >= len(positions):
                    continue
                direction = pristine_pos[j] - vacancy_pos
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue
                positions[j_def] += direction * fraction
        else:
            # Map original site to defected index
            site_def = site
            for vs in sorted(vacancy_sites):
                if vs < site:
                    site_def -= 1
            if site_def < 0 or site_def >= len(positions):
                continue
            defect_pos = positions[site_def]
            n_idx, _ = _get_neighbours(distorted, site_def, cutoff)
            for j in n_idx:
                direction = positions[j] - defect_pos
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue
                positions[j] += direction * fraction

    distorted.set_positions(positions)
    return distorted


def generate_rattle(atoms: Atoms, stdev: float, seed: int) -> Atoms:
    rattled = atoms.copy()
    rng = np.random.default_rng(seed)
    displacements = rng.normal(0, stdev, size=(len(atoms), 3))
    rattled.set_positions(rattled.get_positions() + displacements)
    return rattled


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def _label_with_config(config_name: str, label: str, multi: bool) -> str:
    """Prefix structure labels with config name in multi-config runs.

    In single-config mode (the default), labels stay unchanged so existing
    analysis, tests, and downstream tools that key off label names keep
    working. In multi-config mode, labels become ``<config>__<label>`` so
    they stay unique across configurations in the shared DB.
    """
    return f"{config_name}__{label}" if multi else label


def _generate_one_config(
    defect_config,
    supercell: Atoms,
    db: DefectDB,
    config: Config,
    config_idx: int,
    multi: bool,
) -> None:
    """Apply one defect configuration and generate distortions + rattles."""
    logger.info("--- Defect configuration: %s ---", defect_config.name)

    defected, defect_sites, desc = insert_defects(
        supercell, defect_config.defects,
        seed=config.distortions.seed + 7919 * config_idx,
    )

    # Identify which of the applied defects were vacancies (needed for the
    # bond-distortion helper which treats vacancies specially).
    vacancy_sites: list[int] = []
    idx = 0
    for spec in defect_config.defects:
        for _ in range(spec.count):
            if spec.dtype == "vacancy":
                vacancy_sites.append(defect_sites[idx])
            idx += 1

    base_kvp = dict(
        defect_type=desc,
        defect_config=defect_config.name,
        defect_info=desc,
        n_defects=len(defect_sites),
    )

    # Undistorted reference for this configuration.
    db.add_structure(
        defected.copy(),
        label=_label_with_config(defect_config.name, "undistorted", multi),
        distortion_type="undistorted",
        distortion_mag=0.0,
        **base_kvp,
    )

    # Bond distortions.
    dist_cfg = config.distortions
    for frac in dist_cfg.get_bond_distortions():
        if abs(frac) < 1e-10:
            continue
        distorted = generate_bond_distortion(
            defected, defect_sites, vacancy_sites,
            supercell, defect_sites, frac,
        )
        db.add_structure(
            distorted,
            label=_label_with_config(defect_config.name, f"bond_{frac:+.3f}", multi),
            distortion_type="bond_distortion",
            distortion_mag=frac,
            **base_kvp,
        )

    # Rattles. Seed is offset by config index so different configurations
    # do not draw identical random displacements.
    for i in range(dist_cfg.n_rattle):
        seed = dist_cfg.seed + i + 1000 + 100 * config_idx
        rattled = generate_rattle(defected, dist_cfg.rattle_std, seed)
        db.add_structure(
            rattled,
            label=_label_with_config(defect_config.name, f"rattle_{i}", multi),
            distortion_type="rattle",
            distortion_mag=dist_cfg.rattle_std,
            **base_kvp,
        )


def run_generation(config: Config, db: DefectDB) -> None:
    logger.info("=== Structure Generation ===")

    unit_cell = read(str(config.structure_path))
    logger.info("Unit cell: %s, %d atoms", unit_cell.get_chemical_formula(), len(unit_cell))

    supercell = build_supercell(unit_cell, config)
    db.add_pristine(supercell.copy())

    defect_configs = config.defect_configs
    multi = config.is_multi_config
    logger.info(
        "Generating %d defect configuration(s): %s",
        len(defect_configs),
        ", ".join(c.name for c in defect_configs),
    )

    for idx, dc in enumerate(defect_configs):
        _generate_one_config(dc, supercell, db, config, idx, multi)

    summary = db.summary()
    logger.info(
        "Generation complete: %d structures (%d defect + 1 pristine) across %d config(s)",
        summary["total"], summary["total"] - 1, len(defect_configs),
    )
