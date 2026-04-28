"""Command-line interface for defectool."""

import logging
from pathlib import Path

import click

from . import __version__
from .config import load_config
from .db import DefectDB
from .generate import run_generation
from .relax import run_relaxation, export_xyz
from .analyse import run_analysis


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _export_structures(db, output_dir, logger):
    """Export all structures as POSCAR files."""
    from ase.io import write as ase_write

    struct_dir = output_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    pristine_row = db.get_pristine()
    if pristine_row:
        _, atoms, _ = pristine_row
        ase_write(str(struct_dir / "pristine.vasp"), atoms, format="vasp")

    count = 0
    for row_id, atoms, kvp in db.get_all_defect():
        label = kvp.get("label", f"id_{row_id}")
        safe = label.replace("+", "p").replace("-", "m")
        ase_write(str(struct_dir / f"{safe}.vasp"), atoms, format="vasp")
        count += 1

    logger.info("Exported %d structures to %s/", count + 1, struct_dir)


def _print_summary(db, logger):
    s = db.summary()
    logger.info(
        "Database: total=%d  pristine=%d  generated=%d  "
        "converged=%d  failed=%d",
        s["total"], s["pristine"], s["generated"],
        s["converged"], s["failed"],
    )


@click.group()
@click.version_option(version=__version__)
def cli():
    """defectool: point defect generation, relaxation and analysis."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def run(config_file, verbose):
    """Run full pipeline: generate -> relax -> analyse."""
    _setup_logging(verbose)
    logger = logging.getLogger("defectool")

    config = load_config(config_file)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db = DefectDB(output_dir / config.database)

    modes = config.mode
    logger.info("Modes: %s", modes)

    if "generate" in modes:
        run_generation(config, db)
        _export_structures(db, output_dir, logger)

    if "relax" in modes:
        run_relaxation(config, db)
        export_xyz(config, db)

    if "analyse" in modes:
        converged = db.get_converged()
        if converged:
            run_analysis(config, db)
        else:
            logger.warning("No converged structures. Skipping analysis.")

    _print_summary(db, logger)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def generate(config_file, verbose):
    """Generate defect structures and export as POSCAR files."""
    _setup_logging(verbose)
    logger = logging.getLogger("defectool")

    config = load_config(config_file)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db = DefectDB(output_dir / config.database)

    run_generation(config, db)
    _export_structures(db, output_dir, logger)
    _print_summary(db, logger)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def relax(config_file, verbose):
    """Relax structures via ASE calculator."""
    _setup_logging(verbose)
    logger = logging.getLogger("defectool")

    config = load_config(config_file)
    output_dir = Path(config.output_dir)
    db = DefectDB(output_dir / config.database)

    run_relaxation(config, db)
    export_xyz(config, db)
    _print_summary(db, logger)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def analyse(config_file, verbose):
    """Analyse relaxed structures and generate plots."""
    _setup_logging(verbose)
    logger = logging.getLogger("defectool")

    config = load_config(config_file)
    output_dir = Path(config.output_dir)
    db = DefectDB(output_dir / config.database)

    converged = db.get_converged()
    if not converged:
        logger.error("No converged structures. Run relaxation first.")
        return

    run_analysis(config, db)
    _print_summary(db, logger)


@cli.command()
@click.argument("db_path", type=click.Path(exists=True))
def status(db_path):
    """Print database contents."""
    _setup_logging(False)
    db = DefectDB(db_path)
    s = db.summary()

    click.echo(f"\nDatabase: {db_path}")
    click.echo(f"  Total:     {s['total']}")
    click.echo(f"  Pristine:  {s['pristine']}")
    click.echo(f"  Generated: {s['generated']}")
    click.echo(f"  Converged: {s['converged']}")
    click.echo(f"  Failed:    {s['failed']}")
    click.echo()

    for row_id, atoms, kvp in db.get_all_defect():
        st = kvp.get("status", "?")
        label = kvp.get("label", f"id_{row_id}")
        energy = kvp.get("total_energy")
        e_str = f"{energy:.6f} eV" if energy is not None else "---"
        click.echo(f"  [{st:<10}]  {label:<25}  {e_str}")

    p = db.get_pristine()
    if p:
        _, _, pk = p
        pe = pk.get("total_energy")
        pe_str = f"{pe:.6f} eV" if pe is not None else "---"
        click.echo(f"  [{'pristine':<10}]  {'pristine':<25}  {pe_str}")
    click.echo()


def main():
    cli()


if __name__ == "__main__":
    main()
