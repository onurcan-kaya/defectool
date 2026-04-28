"""End-to-end tests for defectool v0.2."""

import tempfile
from pathlib import Path
import yaml
import logging

from ase.build import bulk
from ase.spacegroup import crystal
from ase.io import write, read
from ase.calculators.lj import LennardJones

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")


def _make_config(tmpdir, structure_file, defects, **overrides):
    cfg = {
        "structure": str(structure_file),
        "defects": defects,
        "supercell": [2, 2, 2],
        "distortions": {
            "bond_distortion_min": -0.1,
            "bond_distortion_max": 0.1,
            "bond_distortion_steps": 5,
            "n_rattle": 2,
            "rattle_std": 0.1,
            "seed": 42,
        },
        "relaxation": {"fmax": 0.1, "max_steps": 30},
        "mode": ["generate", "relax", "analyse"],
        "output_dir": str(tmpdir / "output"),
        "database": "test.db",
    }
    cfg.update(overrides)
    path = tmpdir / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def _run_pipeline(config_path):
    from defectool.config import load_config
    from defectool.db import DefectDB
    from defectool.generate import run_generation
    from defectool.relax import relax_one, export_xyz
    from defectool.analyse import run_analysis

    cfg = load_config(config_path)
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    db = DefectDB(outdir / cfg.database)

    run_generation(cfg, db)

    lj = LennardJones(sigma=2.3, epsilon=0.05)
    for rid, atoms, kvp in db.get_by_status("generated"):
        relax_one(rid, atoms, lj, cfg, db, kvp)

    cfg.analysis.save_format = "png"
    run_analysis(cfg, db)
    export_xyz(cfg, db)

    return cfg, db


def test_single_vacancy():
    print("\n=== Test: Single vacancy (NaCl) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        poscar = tmpdir / "POSCAR"
        write(str(poscar), nacl, format="vasp")

        cfg_path = _make_config(tmpdir, poscar,
                                [{"type": "vacancy", "element": "Na"}])
        cfg, db = _run_pipeline(cfg_path)

        s = db.summary()
        assert s["converged"] > 0, f"Nothing converged: {s}"
        assert (Path(cfg.output_dir) / "relaxed_structures.xyz").exists()
        assert (Path(cfg.output_dir) / "energy_vs_distortion.png").exists()
        print(f"PASS: {s['converged']} converged, plots + xyz exported")


def test_multiple_vacancies():
    print("\n=== Test: Two Na vacancies (NaCl) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        poscar = tmpdir / "POSCAR"
        write(str(poscar), nacl, format="vasp")

        cfg_path = _make_config(tmpdir, poscar,
                                [{"type": "vacancy", "element": "Na", "count": 2}])
        cfg, db = _run_pipeline(cfg_path)

        # Check that 2 atoms were removed
        pristine = db.get_pristine()
        _, p_atoms, _ = pristine
        for _, d_atoms, _ in db.get_all_defect():
            assert len(d_atoms) == len(p_atoms) - 2, \
                f"Expected {len(p_atoms)-2} atoms, got {len(d_atoms)}"
            break
        print(f"PASS: 2 vacancies, {len(p_atoms)-2} atoms in defected cells")


def test_mixed_defects():
    print("\n=== Test: Vacancy + substitution (NaCl) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        poscar = tmpdir / "POSCAR"
        write(str(poscar), nacl, format="vasp")

        cfg_path = _make_config(tmpdir, poscar, [
            {"type": "vacancy", "element": "Na"},
            {"type": "substitution", "element": "Cl", "substitute": "Br"},
        ])
        cfg, db = _run_pipeline(cfg_path)

        for _, atoms, _ in db.get_all_defect():
            symbols = atoms.get_chemical_symbols()
            assert "Br" in symbols, "Br not found"
            assert len(atoms) == 15, f"Expected 15, got {len(atoms)}"
            break
        print(f"PASS: vacancy + substitution, Br present, 15 atoms")


def test_substitution():
    print("\n=== Test: Substitution (Si -> Ge) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        si = bulk("Si", "diamond", a=5.43)
        poscar = tmpdir / "POSCAR"
        write(str(poscar), si, format="vasp")

        cfg_path = _make_config(tmpdir, poscar,
                                [{"type": "substitution", "element": "Si",
                                  "substitute": "Ge"}])
        cfg, db = _run_pipeline(cfg_path)

        for _, atoms, _ in db.get_all_defect():
            assert "Ge" in atoms.get_chemical_symbols()
            break
        print(f"PASS: Si->Ge substitution")


def test_antisite():
    print("\n=== Test: Antisite (GaAs) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        gaas = crystal(
            ["Ga", "As"], basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
            spacegroup=216, cellpar=[5.65]*3 + [90]*3,
        )
        poscar = tmpdir / "POSCAR"
        write(str(poscar), gaas, format="vasp")

        cfg_path = _make_config(tmpdir, poscar,
                                [{"type": "antisite", "element": "Ga"}])
        cfg, db = _run_pipeline(cfg_path)

        s = db.summary()
        assert s["total"] > 1
        print(f"PASS: antisite, {s['total']} structures")


def test_linspace_distortions():
    print("\n=== Test: Linspace distortions ===")
    from defectool.config import DistortionConfig
    dc = DistortionConfig(
        bond_distortion_min=-0.05,
        bond_distortion_max=0.05,
        bond_distortion_steps=5,
    )
    fracs = dc.get_bond_distortions()
    assert len(fracs) == 5
    assert abs(fracs[0] - (-0.05)) < 1e-10
    assert abs(fracs[-1] - 0.05) < 1e-10
    print(f"PASS: linspace gives {fracs}")


def test_extxyz_output():
    print("\n=== Test: Extended XYZ export ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        poscar = tmpdir / "POSCAR"
        write(str(poscar), nacl, format="vasp")

        cfg_path = _make_config(tmpdir, poscar,
                                [{"type": "vacancy", "element": "Na"}])
        cfg, db = _run_pipeline(cfg_path)

        xyz_path = Path(cfg.output_dir) / "relaxed_structures.xyz"
        assert xyz_path.exists(), "XYZ file not created"
        frames = read(str(xyz_path), index=":")
        assert len(frames) > 0
        for frame in frames:
            assert "REF_energy" in frame.info, "Missing energy in info"
            assert "label" in frame.info, "Missing label in info"
        print(f"PASS: {len(frames)} frames in XYZ, all have energy + label")


def test_backward_compat_single_defect():
    """Test that 'defect' (singular) still works in config."""
    print("\n=== Test: Backward compat (singular 'defect' key) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        poscar = tmpdir / "POSCAR"
        write(str(poscar), nacl, format="vasp")

        cfg = {
            "structure": str(poscar),
            "defect": {"type": "vacancy", "element": "Cl"},
            "supercell": [2, 2, 2],
            "mode": ["generate"],
            "output_dir": str(tmpdir / "output"),
        }
        path = tmpdir / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)

        from defectool.config import load_config
        c = load_config(path)
        assert len(c.defects) == 1
        assert c.defects[0].dtype == "vacancy"
        # Plain list (no combinations:) should collapse to a single 'default' config
        assert len(c.defect_configs) == 1
        assert c.defect_configs[0].name == "default"
        assert not c.is_multi_config
        print("PASS: singular 'defect' key parsed correctly, 1 default config")


def test_combinations_expansion():
    """Parse-only: verify the combinations: block produces the right configs."""
    print("\n=== Test: combinations expansion (parse only) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        poscar = tmpdir / "POSCAR"
        write(str(poscar), nacl, format="vasp")

        cfg = {
            "structure": str(poscar),
            "defects": [
                {"type": "vacancy", "element": "Na"},
                {"type": "vacancy", "element": "Cl"},
                {"type": "substitution", "element": "Cl", "substitute": "Br"},
            ],
            "combinations": {
                "singletons": True,
                "pairs": True,
                "include_composite": False,
            },
            "supercell": [2, 2, 2],
            "mode": ["generate"],
            "output_dir": str(tmpdir / "output"),
        }
        path = tmpdir / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)

        from defectool.config import load_config
        c = load_config(path)

        # 3 singletons + C(3,2)=3 pairs = 6 configs
        names = [dc.name for dc in c.defect_configs]
        assert len(names) == 6, f"Expected 6 configs, got {len(names)}: {names}"
        assert "V_Na" in names
        assert "V_Cl" in names
        assert "Cl_to_Br" in names
        assert "V_Na__V_Cl" in names
        assert c.is_multi_config
        print(f"PASS: 6 configs generated — {names}")


def test_combinations_pipeline():
    """End-to-end: Mode B produces one set of outputs per configuration."""
    print("\n=== Test: combinations end-to-end (NaCl singletons) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        poscar = tmpdir / "POSCAR"
        write(str(poscar), nacl, format="vasp")

        cfg_path = _make_config(
            tmpdir, poscar,
            [
                {"type": "vacancy", "element": "Na"},
                {"type": "vacancy", "element": "Cl"},
            ],
            combinations={"singletons": True, "pairs": False},
        )
        cfg, db = _run_pipeline(cfg_path)

        # Exactly two configs should be present in the DB
        configs = db.get_defect_configs()
        assert set(configs) == {"V_Na", "V_Cl"}, f"Got {configs}"

        # Per-config plot files should exist
        outdir = Path(cfg.output_dir)
        for name in ("V_Na", "V_Cl"):
            for stem in ("energy_vs_distortion", "coordination",
                         "bond_lengths", "rdf"):
                f = outdir / f"{stem}_{name}.png"
                assert f.exists(), f"Missing {f}"

        # Legacy single-config filenames should NOT exist in multi-config mode
        assert not (outdir / "energy_vs_distortion.png").exists()

        # Each config should have its own undistorted + bond + rattle rows
        for name in ("V_Na", "V_Cl"):
            rows = [kvp for _, _, kvp in db.get_all_defect()
                    if kvp.get("defect_config") == name]
            labels = {kvp.get("label") for kvp in rows}
            assert any(l.endswith("undistorted") for l in labels), \
                f"No undistorted label for {name}: {labels}"
            assert any("bond_" in l for l in labels), \
                f"No bond distortion for {name}: {labels}"

        # XYZ export should carry defect_config through to info[]
        frames = read(str(outdir / "relaxed_structures.xyz"), index=":")
        cfgs_in_xyz = {f.info.get("defect_config") for f in frames
                       if f.info.get("defect_type") != "pristine"}
        assert cfgs_in_xyz == {"V_Na", "V_Cl"}, f"xyz configs: {cfgs_in_xyz}"

        print(f"PASS: 2 configs, per-config plots + DB tagging + XYZ metadata")


if __name__ == "__main__":
    test_linspace_distortions()
    test_backward_compat_single_defect()
    test_combinations_expansion()
    test_single_vacancy()
    test_substitution()
    test_antisite()
    test_multiple_vacancies()
    test_mixed_defects()
    test_extxyz_output()
    test_combinations_pipeline()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
