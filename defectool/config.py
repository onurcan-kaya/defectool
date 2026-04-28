"""Configuration parsing, defaults and validation."""

from dataclasses import dataclass, field
from itertools import combinations as _iter_combinations
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DefectSpec:
    """Specification of one defect operation."""
    dtype: str                        # vacancy, substitution, antisite
    element: Optional[str] = None     # element name to target
    site: Optional[int] = None        # specific atom index
    substitute: Optional[str] = None  # replacement species (substitution)
    count: int = 1                    # how many of this defect

    def __post_init__(self):
        allowed = ("vacancy", "substitution", "antisite")
        if self.dtype not in allowed:
            raise ValueError(
                f"Defect type must be one of {allowed}, got '{self.dtype}'"
            )
        if self.dtype == "substitution" and self.substitute is None:
            raise ValueError("Substitution defect requires 'substitute'.")
        if self.site is None and self.element is None:
            raise ValueError("Provide either 'site' or 'element'.")


def _spec_tag(spec: "DefectSpec") -> str:
    """Short, filesystem-safe label for a single DefectSpec."""
    target = spec.element if spec.element is not None else f"site{spec.site}"
    if spec.dtype == "vacancy":
        base = f"V_{target}"
    elif spec.dtype == "substitution":
        base = f"{target}_to_{spec.substitute}"
    elif spec.dtype == "antisite":
        base = f"AS_{target}"
    else:
        base = spec.dtype
    if spec.count > 1:
        base = f"{base}x{spec.count}"
    return base


@dataclass
class DefectConfig:
    """A named bundle of DefectSpecs that together form one defect configuration.

    When ``combinations:`` is not used, the whole ``defects:`` list lives in
    a single DefectConfig (named ``default``) and everything behaves as
    before. When ``combinations:`` is used, the pool is expanded into
    multiple DefectConfigs and each is processed independently.
    """
    name: str
    defects: list["DefectSpec"]

    @classmethod
    def from_specs(cls, specs: list["DefectSpec"], name: Optional[str] = None) -> "DefectConfig":
        if not specs:
            raise ValueError("DefectConfig requires at least one DefectSpec.")
        if name is None:
            name = "__".join(_spec_tag(s) for s in specs)
        return cls(name=name, defects=list(specs))


@dataclass
class CombinationsConfig:
    """Enumeration of defect configurations from a pool of DefectSpecs.

    With N pool entries:
      * singletons -> N configurations, one per pool entry
      * pairs      -> C(N, 2) configurations, each applying two pool entries together
      * triples    -> C(N, 3) configurations
      * include_composite -> one configuration applying *all* pool entries at once
    """
    singletons: bool = True
    pairs: bool = False
    triples: bool = False
    include_composite: bool = False


@dataclass
class CalculatorConfig:
    name: str = "mace_mp"
    model: str = "medium"
    device: str = "cpu"
    # LAMMPS (ASE LAMMPSlib)
    pair_style: Optional[str] = None
    pair_coeff: Optional[list] = None
    # QE (ASE Espresso)
    pseudopotentials: Optional[dict] = None
    pseudo_dir: Optional[str] = None
    ecutwfc: float = 60.0
    ecutrho: float = 480.0
    kspacing: float = 0.04
    smearing: str = "cold"
    degauss: float = 0.01


@dataclass
class DistortionConfig:
    rattle_std: float = 0.15
    n_rattle: int = 3
    seed: int = 42
    # Bond distortions via range (preferred)
    bond_distortion_min: float = -0.1
    bond_distortion_max: float = 0.1
    bond_distortion_steps: int = 10
    # Or explicit list (overrides range if set)
    bond_distortions: Optional[list] = None

    def get_bond_distortions(self) -> list[float]:
        """Return the list of bond distortion fractions."""
        if self.bond_distortions is not None:
            return self.bond_distortions
        values = np.linspace(
            self.bond_distortion_min,
            self.bond_distortion_max,
            self.bond_distortion_steps,
        ).tolist()
        return values


@dataclass
class RelaxConfig:
    fmax: float = 0.03
    optimizer: str = "FIRE"
    max_steps: int = 1000
    restart: bool = True


@dataclass
class AnalysisConfig:
    rdf_rmax: float = 8.0
    rdf_nbins: int = 200
    coordination_cutoff: Optional[float] = None
    save_format: str = "pdf"


@dataclass
class Config:
    """Top-level configuration."""
    structure_path: Path
    defects: list[DefectSpec]
    # Expanded list of defect configurations. When combinations is not used
    # this contains exactly one entry with name='default' holding all of
    # `defects`. When combinations is used, each entry is one enumerated
    # configuration (singletons / pairs / triples / composite).
    defect_configs: list[DefectConfig] = field(default_factory=list)
    combinations: Optional[CombinationsConfig] = None
    supercell: Optional[list] = None
    min_cell_length: float = 10.0
    calculator: CalculatorConfig = field(default_factory=CalculatorConfig)
    distortions: DistortionConfig = field(default_factory=DistortionConfig)
    relaxation: RelaxConfig = field(default_factory=RelaxConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    mode: list = field(default_factory=lambda: ["generate", "relax", "analyse"])
    database: str = "defectool.db"
    output_dir: str = "defectool_output"
    nprocs: int = 1

    @property
    def is_multi_config(self) -> bool:
        """True when more than one defect configuration is being generated."""
        return len(self.defect_configs) > 1


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(defaults: dict, overrides: dict) -> dict:
    merged = dict(defaults)
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


_CALC_DEFAULTS = {
    "name": "mace_mp", "model": "medium", "device": "cpu",
    "pair_style": None, "pair_coeff": None,
    "pseudopotentials": None, "pseudo_dir": None,
    "ecutwfc": 60.0, "ecutrho": 480.0, "kspacing": 0.04,
    "smearing": "cold", "degauss": 0.01,
}

_DIST_DEFAULTS = {
    "rattle_std": 0.15, "n_rattle": 3, "seed": 42,
    "bond_distortion_min": -0.1, "bond_distortion_max": 0.1,
    "bond_distortion_steps": 10, "bond_distortions": None,
}

_RELAX_DEFAULTS = {
    "fmax": 0.03, "optimizer": "FIRE", "max_steps": 1000, "restart": True,
}

_ANALYSIS_DEFAULTS = {
    "rdf_rmax": 8.0, "rdf_nbins": 200,
    "coordination_cutoff": None, "save_format": "pdf",
}


_COMB_DEFAULTS = {
    "singletons": True,
    "pairs": False,
    "triples": False,
    "include_composite": False,
}


def _parse_defects(raw) -> list[DefectSpec]:
    """Parse defect specification. Accepts a single dict or a list of dicts."""
    if isinstance(raw, dict):
        raw = [raw]
    defects = []
    for d in raw:
        defects.append(DefectSpec(
            dtype=d["type"],
            element=d.get("element"),
            site=d.get("site"),
            substitute=d.get("substitute"),
            count=d.get("count", 1),
        ))
    return defects


def _expand_defect_configs(
    pool: list[DefectSpec],
    combinations: Optional[CombinationsConfig],
) -> list[DefectConfig]:
    """Expand a pool of DefectSpecs into a list of DefectConfigs.

    If ``combinations`` is None, return a single 'default' config containing
    all pool entries (current behaviour). Otherwise enumerate according to
    the requested singletons / pairs / triples / composite flags.
    """
    if combinations is None:
        # Current behaviour: treat the list as one composite configuration.
        return [DefectConfig.from_specs(pool, name="default")]

    configs: list[DefectConfig] = []
    seen_names: set[str] = set()

    def _add(specs: list[DefectSpec]):
        cfg = DefectConfig.from_specs(specs)
        if cfg.name in seen_names:
            return
        seen_names.add(cfg.name)
        configs.append(cfg)

    if combinations.singletons:
        for s in pool:
            _add([s])

    if combinations.pairs:
        if len(pool) < 2:
            raise ValueError(
                "combinations.pairs=true requires at least 2 entries in the defect pool."
            )
        for combo in _iter_combinations(pool, 2):
            _add(list(combo))

    if combinations.triples:
        if len(pool) < 3:
            raise ValueError(
                "combinations.triples=true requires at least 3 entries in the defect pool."
            )
        for combo in _iter_combinations(pool, 3):
            _add(list(combo))

    if combinations.include_composite:
        _add(list(pool))

    if not configs:
        raise ValueError(
            "combinations: block enabled but no configurations were produced. "
            "Enable at least one of singletons / pairs / triples / include_composite."
        )

    return configs


def load_config(path: str | Path) -> Config:
    """Load YAML config and return a validated Config object."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if "structure" not in raw:
        raise ValueError("Config must specify 'structure'.")
    if "defects" not in raw and "defect" not in raw:
        raise ValueError("Config must specify 'defects' (or 'defect').")

    # Accept both 'defect' (single) and 'defects' (list)
    defect_raw = raw.get("defects", raw.get("defect"))
    defects = _parse_defects(defect_raw)

    combinations = None
    if "combinations" in raw and raw["combinations"] is not None:
        comb_raw = _deep_merge(_COMB_DEFAULTS, raw["combinations"])
        combinations = CombinationsConfig(**comb_raw)

    defect_configs = _expand_defect_configs(defects, combinations)

    calc_raw = _deep_merge(_CALC_DEFAULTS, raw.get("calculator", {}))
    dist_raw = _deep_merge(_DIST_DEFAULTS, raw.get("distortions", {}))
    relax_raw = _deep_merge(_RELAX_DEFAULTS, raw.get("relaxation", {}))
    analysis_raw = _deep_merge(_ANALYSIS_DEFAULTS, raw.get("analysis", {}))

    config = Config(
        structure_path=Path(raw["structure"]),
        defects=defects,
        defect_configs=defect_configs,
        combinations=combinations,
        supercell=raw.get("supercell"),
        min_cell_length=raw.get("min_cell_length", 10.0),
        calculator=CalculatorConfig(**calc_raw),
        distortions=DistortionConfig(**dist_raw),
        relaxation=RelaxConfig(**relax_raw),
        analysis=AnalysisConfig(**analysis_raw),
        mode=raw.get("mode", ["generate", "relax", "analyse"]),
        database=raw.get("database", "defectool.db"),
        output_dir=raw.get("output_dir", "defectool_output"),
        nprocs=raw.get("nprocs", 1),
    )

    _validate(config)
    return config


def _validate(config: Config):
    if not config.structure_path.exists():
        raise FileNotFoundError(
            f"Structure file not found: {config.structure_path}"
        )
    valid_modes = {"generate", "relax", "analyse"}
    for m in config.mode:
        if m not in valid_modes:
            raise ValueError(f"Unknown mode '{m}'.")

    valid_calcs = {"mace_mp", "mace", "lammps", "espresso"}
    if config.calculator.name not in valid_calcs:
        raise ValueError(f"Unknown calculator '{config.calculator.name}'.")

    if config.calculator.name == "lammps":
        if not config.calculator.pair_style:
            raise ValueError("LAMMPS requires 'pair_style'.")
        if not config.calculator.pair_coeff:
            raise ValueError("LAMMPS requires 'pair_coeff'.")

    if config.calculator.name == "espresso":
        if not config.calculator.pseudopotentials:
            raise ValueError("QE requires 'pseudopotentials'.")

    if config.supercell is not None and len(config.supercell) != 3:
        raise ValueError("Supercell must be 3 integers.")

    if not config.defect_configs:
        raise ValueError("No defect configurations to run.")

    names = [cfg.name for cfg in config.defect_configs]
    if len(set(names)) != len(names):
        dup = [n for n in names if names.count(n) > 1]
        raise ValueError(
            f"Duplicate defect configuration name(s): {sorted(set(dup))}"
        )
