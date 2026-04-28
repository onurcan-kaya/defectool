"""Analysis: energy landscape, coordination, bonding, RDF."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase import Atoms
from ase.neighborlist import neighbor_list

from .config import Config
from .db import DefectDB

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def _collect_energies(
    db: DefectDB, defect_config: Optional[str] = None,
) -> tuple[list, Optional[float]]:
    """Collect relaxed defect entries and reference energy.

    Each entry includes is_converged and final_fmax so analysis can
    separate properly converged structures from those that hit max_steps.

    If ``defect_config`` is given, only entries with that config name are
    returned. Entries written by older versions of defectool (without a
    ``defect_config`` kvp) are treated as belonging to config ``'default'``.
    """
    rows = db.get_converged()
    ref_energy = None
    entries = []

    for row_id, atoms, kvp in rows:
        if kvp.get("defect_type") == "pristine":
            continue
        if defect_config is not None:
            row_cfg = kvp.get("defect_config", "default")
            if row_cfg != defect_config:
                continue
        energy = kvp.get("total_energy")
        if energy is None:
            continue

        is_conv = bool(kvp.get("is_converged", 1))
        fmax = kvp.get("final_fmax", 0.0)

        if kvp.get("distortion_type") == "undistorted":
            ref_energy = energy
        entries.append({
            "id": row_id, "atoms": atoms, "energy": energy,
            "label": kvp.get("label", ""), "dist_type": kvp.get("distortion_type", ""),
            "dist_mag": kvp.get("distortion_mag", 0.0),
            "converged": is_conv, "final_fmax": fmax,
            "defect_config": kvp.get("defect_config", "default"),
        })

    entries.sort(key=lambda x: x["energy"])
    return entries, ref_energy


# ---------------------------------------------------------------------------
# Energy vs distortion
# ---------------------------------------------------------------------------

def _outfile(output_dir: Path, stem: str, fmt: str, defect_config: Optional[str]) -> Path:
    """Build the output path, adding a ``_<config>`` suffix when requested."""
    if defect_config:
        return output_dir / f"{stem}_{defect_config}.{fmt}"
    return output_dir / f"{stem}.{fmt}"


def _title_suffix(defect_config: Optional[str]) -> str:
    return f" — {defect_config}" if defect_config else ""


def plot_energy_landscape(
    db: DefectDB, output_dir: Path, fmt: str = "pdf",
    defect_config: Optional[str] = None,
) -> Optional[Path]:
    entries, ref_energy = _collect_energies(db, defect_config=defect_config)
    # Only keep converged structures
    entries = [e for e in entries if e["converged"]]
    if not entries or ref_energy is None:
        logger.warning("Not enough converged data for energy landscape plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    bond_x, bond_y = [], []
    rattle_y = []

    for e in entries:
        rel = e["energy"] - ref_energy
        if e["dist_type"] == "bond_distortion":
            bond_x.append(e["dist_mag"])
            bond_y.append(rel)
        elif e["dist_type"] == "rattle":
            rattle_y.append(rel)

    if bond_x:
        order = np.argsort(bond_x)
        bx = [bond_x[i] for i in order]
        by = [bond_y[i] for i in order]
        ax.plot(bx, by, "o-", color="#2563eb", markersize=7, linewidth=1.5,
                label="Bond distortions")
        for x, y in zip(bx, by):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, ha="center", color="#2563eb")

    if rattle_y:
        x_rattle = np.zeros(len(rattle_y))
        ax.scatter(x_rattle, rattle_y, marker="s", color="#dc2626", s=60,
                   zorder=5, label="Rattled")
        for i, y in enumerate(rattle_y):
            ax.annotate(f"{y:.3f}", (0, y), textcoords="offset points",
                        xytext=(15, 0), fontsize=7, color="#dc2626")

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    gs = entries[0]  # already sorted, and now all entries are converged
    gs_rel = gs["energy"] - ref_energy
    if gs_rel < -0.005:
        ax.annotate(
            f"Ground state: {gs['label']}\n{gs_rel:.4f} eV",
            xy=(gs.get("dist_mag", 0), gs_rel),
            xytext=(0.3, 0.9), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#dc2626"),
            fontsize=9, color="#dc2626", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#dc2626", alpha=0.9),
        )

    ax.set_xlabel("Bond distortion fraction")
    ax.set_ylabel("Energy relative to undistorted (eV)")
    ax.set_title(f"Energy vs distortion{_title_suffix(defect_config)}")
    ax.legend(frameon=False)

    outpath = _outfile(output_dir, "energy_vs_distortion", fmt, defect_config)
    fig.savefig(outpath)
    plt.close(fig)
    logger.info("Saved: %s", outpath)
    return outpath


# ---------------------------------------------------------------------------
# Coordination
# ---------------------------------------------------------------------------

def plot_coordination(db: DefectDB, output_dir: Path,
                      cutoff: Optional[float] = None,
                      fmt: str = "pdf",
                      defect_config: Optional[str] = None) -> Optional[Path]:
    pristine_row = db.get_pristine()
    if pristine_row is None:
        return None

    _, pristine, _ = pristine_row

    if cutoff is None:
        _, _, d = neighbor_list("ijd", pristine, cutoff=5.0)
        cutoff = 1.3 * np.min(d[d > 0.5]) if len(d) > 0 else 3.0
        logger.info("Auto coordination cutoff: %.2f A", cutoff)

    entries, ref_energy = _collect_energies(db, defect_config=defect_config)
    if not entries:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coordination number distribution for ground state vs undistorted
    undistorted = None
    gs = _get_ground_state(entries)

    for entry, color, label_prefix in [
        (next((e for e in entries if e["dist_type"] == "undistorted"), None),
         "#2563eb", "Undistorted"),
        (gs, "#dc2626", "Ground state"),
    ]:
        if entry is None:
            continue
        atoms = entry["atoms"]
        i_arr, j_arr, _ = neighbor_list("ijd", atoms, cutoff=cutoff)
        cn = np.bincount(i_arr, minlength=len(atoms))
        unique, counts = np.unique(cn, return_counts=True)

        ax = axes[0]
        ax.bar(unique + (0.2 if "Ground" in label_prefix else -0.2),
               counts, width=0.35, color=color, alpha=0.7,
               label=f"{label_prefix} ({entry['label']})")

    axes[0].set_xlabel("Coordination number")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"Coordination distribution (cutoff={cutoff:.2f} A)"
        f"{_title_suffix(defect_config)}"
    )
    axes[0].legend(frameon=False, fontsize=9)

    # Average coordination vs distortion
    avg_cn = []
    dist_mags = []
    for entry in entries:
        if entry["dist_type"] != "bond_distortion":
            continue
        atoms = entry["atoms"]
        i_arr, _, _ = neighbor_list("ijd", atoms, cutoff=cutoff)
        cn = np.bincount(i_arr, minlength=len(atoms))
        avg_cn.append(np.mean(cn))
        dist_mags.append(entry["dist_mag"])

    if avg_cn:
        order = np.argsort(dist_mags)
        axes[1].plot([dist_mags[i] for i in order],
                     [avg_cn[i] for i in order],
                     "o-", color="#2563eb", markersize=6)
    axes[1].set_xlabel("Bond distortion fraction")
    axes[1].set_ylabel("Average coordination number")
    axes[1].set_title(f"Coordination vs distortion{_title_suffix(defect_config)}")

    outpath = _outfile(output_dir, "coordination", fmt, defect_config)
    fig.savefig(outpath)
    plt.close(fig)
    logger.info("Saved: %s", outpath)
    return outpath


# ---------------------------------------------------------------------------
# Bond lengths
# ---------------------------------------------------------------------------

def plot_bond_lengths(db: DefectDB, output_dir: Path,
                      cutoff: float = 3.5, fmt: str = "pdf",
                      defect_config: Optional[str] = None) -> Optional[Path]:
    pristine_row = db.get_pristine()
    if pristine_row is None:
        return None
    _, pristine, _ = pristine_row

    entries, ref_energy = _collect_energies(db, defect_config=defect_config)
    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.5, cutoff, 80)

    # Pristine
    _, _, d_p = neighbor_list("ijd", pristine, cutoff=cutoff)
    ax.hist(d_p, bins=bins, alpha=0.3, color="grey", label="Pristine",
            density=True)

    # Undistorted
    for e in entries:
        if e["dist_type"] == "undistorted":
            _, _, d = neighbor_list("ijd", e["atoms"], cutoff=cutoff)
            ax.hist(d, bins=bins, alpha=0.6, color="#2563eb",
                    label="Undistorted", density=True, histtype="step",
                    linewidth=1.5)
            break

    # Ground state (converged only)
    gs = _get_ground_state(entries)
    if gs is not None and gs["dist_type"] != "undistorted":
        _, _, d_gs = neighbor_list("ijd", gs["atoms"], cutoff=cutoff)
        rel_e = gs["energy"] - ref_energy if ref_energy else 0
        ax.hist(d_gs, bins=bins, alpha=0.6, color="#dc2626",
                label=f"GS: {gs['label']} ({rel_e:.3f} eV)",
                density=True, histtype="step", linewidth=1.5)

    ax.set_xlabel("Bond length (A)")
    ax.set_ylabel("Density")
    ax.set_title(f"Bond length distribution{_title_suffix(defect_config)}")
    ax.legend(frameon=False, fontsize=9)

    outpath = _outfile(output_dir, "bond_lengths", fmt, defect_config)
    fig.savefig(outpath)
    plt.close(fig)
    logger.info("Saved: %s", outpath)
    return outpath


# ---------------------------------------------------------------------------
# RDF
# ---------------------------------------------------------------------------

def compute_rdf(atoms, rmax=8.0, nbins=200):
    _, _, distances = neighbor_list("ijd", atoms, cutoff=rmax)
    hist, edges = np.histogram(distances, bins=nbins, range=(0.01, rmax))
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    n = len(atoms)
    rho = n / atoms.get_volume()
    shell_vol = 4 * np.pi * r**2 * dr
    g_r = hist / (n * rho * shell_vol)
    return r, g_r


def plot_rdf(db: DefectDB, output_dir: Path,
             rmax: float = 8.0, nbins: int = 200,
             fmt: str = "pdf",
             defect_config: Optional[str] = None) -> Optional[Path]:
    pristine_row = db.get_pristine()
    if pristine_row is None:
        return None
    _, pristine, _ = pristine_row

    entries, ref_energy = _collect_energies(db, defect_config=defect_config)
    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    r, g = compute_rdf(pristine, rmax, nbins)
    ax.plot(r, g, color="grey", alpha=0.7, label="Pristine", linewidth=1.2)

    for e in entries:
        if e["dist_type"] == "undistorted":
            r2, g2 = compute_rdf(e["atoms"], rmax, nbins)
            ax.plot(r2, g2, color="#2563eb", label="Undistorted", linewidth=1.2)
            break

    gs = _get_ground_state(entries)
    if gs is not None and gs["dist_type"] != "undistorted":
        r3, g3 = compute_rdf(gs["atoms"], rmax, nbins)
        rel_e = gs["energy"] - ref_energy if ref_energy else 0
        ax.plot(r3, g3, color="#dc2626",
                label=f"GS: {gs['label']} ({rel_e:.3f} eV)", linewidth=1.2)

    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.set_title(f"Radial distribution function{_title_suffix(defect_config)}")
    ax.legend(frameon=False, fontsize=9)

    outpath = _outfile(output_dir, "rdf", fmt, defect_config)
    fig.savefig(outpath)
    plt.close(fig)
    logger.info("Saved: %s", outpath)
    return outpath


# ---------------------------------------------------------------------------
# Energy summary table
# ---------------------------------------------------------------------------

def _get_ground_state(entries: list) -> Optional[dict]:
    """Return the lowest-energy converged entry, or None."""
    converged = [e for e in entries if e["converged"]]
    if not converged:
        return None
    converged.sort(key=lambda x: x["energy"])
    return converged[0]


def print_energy_summary(db: DefectDB, defect_config: Optional[str] = None) -> None:
    entries, ref_energy = _collect_energies(db, defect_config=defect_config)
    if not entries:
        logger.warning("No relaxed structures%s.",
                       f" for config '{defect_config}'" if defect_config else "")
        return

    gs = _get_ground_state(entries)

    n_unconv = sum(1 for e in entries if not e["converged"])

    # Sort: converged first (by energy), unconverged at the bottom
    converged_sorted = sorted(
        [e for e in entries if e["converged"]], key=lambda x: x["energy"]
    )
    unconverged = [e for e in entries if not e["converged"]]
    entries = converged_sorted + unconverged

    header = "=" * 90
    if defect_config:
        print(f"\n{header}")
        print(f"Defect configuration: {defect_config}")
        print(header)
    else:
        print(f"\n{header}")

    print(f"{'Label':<30} {'Type':<18} {'Mag':>8} {'E (eV)':>14} "
          f"{'dE (eV)':>10} {'fmax':>8} {'Status':>10}")
    print("-" * 90)

    for e in entries:
        if not e["converged"]:
            print(f"{e['label']:<30} {e['dist_type']:<18} {e['dist_mag']:>8.4f} "
                  f"{'---':>14} {'---':>10} {'---':>8} {'UNCONVERGED':>15}")
            continue
        de = e["energy"] - ref_energy if ref_energy else 0
        fmax_str = f"{e['final_fmax']:.4f}" if e["final_fmax"] else "---"
        if gs is not None and e is gs:
            status = "<-- GS"
        else:
            status = "ok"
        print(f"{e['label']:<30} {e['dist_type']:<18} {e['dist_mag']:>8.4f} "
              f"{e['energy']:>14.6f} {de:>10.4f} {fmax_str:>8} {status:>15}")

    print("=" * 90)

    if n_unconv > 0:
        print(f"\n  {n_unconv} structure(s) did not converge and are excluded "
              f"from ground state selection.")

    if gs is not None and ref_energy is not None:
        de = gs["energy"] - ref_energy
        if de < -0.001:
            print(f"\n  Reconstruction found: {gs['label']} "
                  f"({de:.4f} eV below undistorted)")
        else:
            print("\n  No energy-lowering reconstruction found.")
    else:
        print("\n  No converged structures to determine ground state.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(config: Config, db: DefectDB) -> None:
    logger.info("=== Analysis ===")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = config.analysis.save_format

    # Figure out which configs actually have data in the DB. Fall back to
    # whatever is in the Config object (useful if analyse is run before
    # any relaxation produced a converged row — in that case we produce no
    # plots but still don't crash).
    configs_in_db = db.get_defect_configs()
    if not configs_in_db:
        configs_in_db = [c.name for c in config.defect_configs] or ["default"]

    multi = len(configs_in_db) > 1

    if not multi:
        # Single-config: preserve legacy output filenames (no config suffix).
        print_energy_summary(db, defect_config=None)
        plot_energy_landscape(db, output_dir, fmt)
        plot_coordination(db, output_dir,
                          cutoff=config.analysis.coordination_cutoff, fmt=fmt)
        plot_bond_lengths(db, output_dir, fmt=fmt)
        plot_rdf(db, output_dir,
                 rmax=config.analysis.rdf_rmax,
                 nbins=config.analysis.rdf_nbins, fmt=fmt)
    else:
        # Multi-config: one set of plots + one table per defect configuration.
        logger.info("Analysing %d defect configurations: %s",
                    len(configs_in_db), ", ".join(configs_in_db))
        for cfg_name in configs_in_db:
            print_energy_summary(db, defect_config=cfg_name)
            plot_energy_landscape(db, output_dir, fmt, defect_config=cfg_name)
            plot_coordination(db, output_dir,
                              cutoff=config.analysis.coordination_cutoff,
                              fmt=fmt, defect_config=cfg_name)
            plot_bond_lengths(db, output_dir, fmt=fmt, defect_config=cfg_name)
            plot_rdf(db, output_dir,
                     rmax=config.analysis.rdf_rmax,
                     nbins=config.analysis.rdf_nbins,
                     fmt=fmt, defect_config=cfg_name)

    logger.info("Analysis complete. Outputs in %s/", output_dir)
