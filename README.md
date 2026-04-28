# defectool

A command-line tool for generating, relaxing and analysing point defects in crystalline materials. Given a unit cell and one or more defect specifications, defectool builds a supercell, introduces the defects, generates a set of distorted structures (bond distortions and random rattles), relaxes them through an ASE calculator, and identifies the lowest-energy reconstruction. The approach systematically explores local reconstructions that may lower the energy.

## Installation

```bash
git clone https://github.com/onurcan-kaya/defectool.git
cd defectool
pip install -e .
```

For the MACE-MP foundation model (recommended default calculator):

```bash
pip install -e ".[mace]"
```

This installs `mace-torch` which pulls in PyTorch (large download).

### Dependencies

Required: Python >= 3.10, ASE, spglib, NumPy, SciPy, Matplotlib, Click, PyYAML.

Optional: mace-torch (MACE calculators), LAMMPS Python bindings (for classical potentials through ASE), Quantum ESPRESSO with a working `pw.x` binary (for DFT through ASE).

## Quick start

Create a YAML config file, then run:

```bash
defectool run config.yaml
```

This executes all three stages: generate -> relax -> analyse. You can also run them individually:

```bash
defectool generate config.yaml   # create structures only
defectool relax config.yaml      # relax existing structures in the database
defectool analyse config.yaml    # generate plots from relaxed data
defectool status output/defectool.db   # print database contents
```

## How it works

**Generation.** Reads the unit cell, builds a supercell (automatically if not specified), identifies symmetry-inequivalent defect sites via spglib, and applies defects. For each defect configuration it creates an undistorted reference, a set of bond-distorted variants (nearest-neighbour bonds around the defect site stretched or compressed by controlled fractions), and a set of randomly rattled variants. Everything is stored in an ASE SQLite database and exported as POSCAR files.

**Relaxation.** Each structure is relaxed using the configured ASE calculator and optimiser. The relaxation loop runs step by step and aborts early if the energy diverges by more than 10x the initial magnitude (catches the model extrapolating outside its training range after aggressive rattles or bond compressions). Each structure is marked converged or unconverged based on the final fmax. Results are written to the database and exported as a single extended XYZ file with energy and metadata in the info dict.

**Analysis.** Reads converged energies, prints a ranked summary table, and produces four plots. Unconverged structures are shown in the table marked UNCONVERGED with dashes instead of energy values, and they are excluded from the plots and from ground state selection.

**Parallel execution.** If `nprocs > 1`, relaxations run in parallel via `ProcessPoolExecutor`. The calculator is built once per worker process (important for MACE where model loading is expensive), and structures are distributed across workers.

## Configuration reference

Only `structure` and `defects` are required. Everything else has defaults.

### structure

Path to the unit cell file. Any format ASE can read: CIF, POSCAR, extxyz, etc.

```yaml
structure: BN.poscar
```

### defects

A list of defect operations. Each entry has a `type`, a target (`element` or `site`), and optionally `substitute` and `count`.

Defect types:

- `vacancy` removes an atom.
- `substitution` replaces an atom with a different species. Requires `substitute`.
- `antisite` swaps an atom with its nearest neighbour of a different species.

Target selection:

- `element: N` targets all atoms of that element. The tool uses spglib to find symmetry-inequivalent sites and picks one representative per group.
- `site: 12` targets a specific atom index in the supercell.

The `count` field (default 1) controls how many defects of that type to create. For count > 1, sites are chosen randomly with a fixed seed for reproducibility.

You can also use `defect:` (singular) for a single defect instead of a list.

By default, every entry in `defects:` is applied to the **same** supercell (so e.g. `[V_N, V_B]` produces a divacancy structure). To run defectool over a *pool* of defects and explore each combination as its own configuration, see [combinations](#combinations) below.

### combinations

Optional. Turns the `defects:` list into a *pool* and enumerates configurations from it. Without this block, defectool keeps its original behaviour (apply all listed defects together to one structure).

```yaml
defects:
  - type: vacancy
    element: N
  - type: vacancy
    element: B
  - type: substitution
    element: B
    substitute: O

combinations:
  singletons: true          # one config per pool entry          (default true)
  pairs: false              # all 2-combinations from the pool   (default false)
  triples: false            # all 3-combinations from the pool   (default false)
  include_composite: false  # one config applying every entry    (default false)
```

With the example above and `singletons: true, pairs: true`, defectool generates **six** independent defect configurations: `V_N`, `V_B`, `B_to_O`, `V_N__V_B`, `V_N__B_to_O`, `V_B__B_to_O`. Each runs through its own bond-distortion + rattle grid against the same shared supercell and pristine reference.

In multi-config mode:

- Structure labels are prefixed with the configuration name (e.g. `V_N__bond_-0.050`) so they remain unique inside the shared database.
- Every database row carries a `defect_config=<name>` key, also exposed in `info["defect_config"]` of the exported extended XYZ.
- Plots and summary tables are produced once per configuration: `energy_vs_distortion_V_N.pdf`, `rdf_V_B.pdf`, etc.
- Single-config behaviour (no `combinations:` block, or only one configuration produced) is unchanged: legacy filenames like `energy_vs_distortion.pdf` are preserved.

This is the right mode for building diverse training datasets for MLIPs (e.g. via [`defectset`](https://github.com/onurcan-kaya/defectset)), where you want many distinct defect environments rather than one composite supercell.

### supercell

Either explicit repetitions or a minimum cell length:

```yaml
supercell: [3, 3, 1]         # explicit
# or
min_cell_length: 10.0         # auto (default)
```

Auto mode determines repetitions so every cell vector is at least `min_cell_length` angstrom. Default is 10 angstrom.

### calculator

All calculators run through ASE.

**mace_mp** (default). The MACE-MP-0 foundation model. Works for any element, no extra files needed. Requires mace-torch.

```yaml
calculator:
  name: mace_mp
  model: medium          # small, medium, large
  device: cpu             # cpu or cuda
```

**mace**. Custom MACE model from a local checkpoint.

```yaml
calculator:
  name: mace
  model: /path/to/model.model
  device: cpu
```

**lammps**. Classical potentials through ASE's LAMMPSlib. Requires LAMMPS built with Python bindings.

```yaml
calculator:
  name: lammps
  pair_style: "tersoff"
  pair_coeff: ["* * BNC.tersoff B N C"]
```

**espresso**. Quantum ESPRESSO through ASE's Espresso interface. Requires `pw.x` in PATH.

```yaml
calculator:
  name: espresso
  pseudopotentials:
    B: B.pbe-n-kjpaw_psl.1.0.0.UPF
    N: N.pbe-n-kjpaw_psl.1.0.0.UPF
  pseudo_dir: ./pseudo
  ecutwfc: 60.0              # default
  ecutrho: 480.0             # default
  kspacing: 0.04             # default
  smearing: cold             # default
  degauss: 0.01              # default
```

### distortions

**Bond distortions** stretch or compress nearest-neighbour bonds around each defect site by a fractional amount. A fraction of -0.1 compresses bonds by 10%, +0.1 stretches them by 10%.

Specify a range (preferred, uses `numpy.linspace` internally):

```yaml
distortions:
  bond_distortion_min: -0.1        # default
  bond_distortion_max: 0.1         # default
  bond_distortion_steps: 10        # default
```

Or an explicit list (overrides the range):

```yaml
distortions:
  bond_distortions: [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
```

**Rattle** applies random Gaussian displacements to all atoms.

```yaml
distortions:
  rattle_std: 0.15       # stdev in angstrom (default)
  n_rattle: 3            # number of rattled structures (default)
  seed: 42               # random seed (default)
```

For materials with short bonds (e.g. hBN, graphene) use a smaller `rattle_std` like 0.05 to avoid atoms overlapping.

### relaxation

```yaml
relaxation:
  fmax: 0.03             # force convergence (eV/angstrom), default
  optimizer: FIRE         # FIRE, BFGS or LBFGS (default: FIRE)
  max_steps: 1000         # default
  restart: true           # retry failed structures on re-run (default)
```

**Optimiser choices.** FIRE is the default and best for distorted defects (handles large initial forces well). BFGS can be faster when starting close to a minimum. LBFGS uses less memory for very large cells.

### analysis

```yaml
analysis:
  rdf_rmax: 8.0                    # RDF cutoff in angstrom (default)
  rdf_nbins: 200                    # (default)
  coordination_cutoff: null          # null = auto from pristine (default)
  save_format: pdf                   # pdf, png or svg (default: pdf)
```

### mode

Which stages to run:

```yaml
mode: [generate, relax, analyse]   # default
```

Any subset: `[generate]`, `[generate, relax]`, `[analyse]`, etc.

### nprocs

Number of parallel processes for relaxation. Default is 1 (serial).

```yaml
nprocs: 4
```

For MACE on GPU, keep `nprocs: 1` (GPU is the bottleneck). For MACE on CPU or LAMMPS, set this to your physical core count. Also set `OMP_NUM_THREADS=1` in your shell when using `nprocs > 1` to avoid thread oversubscription:

```bash
export OMP_NUM_THREADS=1
defectool run config.yaml
```

### output

```yaml
output_dir: defectool_output       # default
database: defectool.db              # default
```

## Recipes

### Single nitrogen vacancy in hBN

```yaml
structure: BN.poscar
defects:
  - type: vacancy
    element: N
supercell: [4, 4, 2]
calculator:
  name: mace_mp
  model: medium
distortions:
  bond_distortion_min: -0.05
  bond_distortion_max: 0.05
  bond_distortion_steps: 10
  rattle_std: 0.05
  n_rattle: 3
relaxation:
  fmax: 0.01
  optimizer: FIRE
  max_steps: 1000
mode: [generate, relax, analyse]
nprocs: 4
```

### 1 B vacancy + 1 N vacancy in hBN

```yaml
structure: BN.poscar
defects:
  - type: vacancy
    element: B
  - type: vacancy
    element: N
supercell: [4, 4, 2]
calculator:
  name: mace_mp
distortions:
  bond_distortion_min: -0.05
  bond_distortion_max: 0.05
  bond_distortion_steps: 10
  rattle_std: 0.05
  n_rattle: 3
mode: [generate, relax, analyse]
nprocs: 4
```

### Diverse defect set for MLIP training (hBN, combinations mode)

Treats the `defects:` list as a pool. With `singletons: true` and `pairs: true`, this produces 5 single-defect configurations and 10 paired configurations (15 independent defect landscapes) from a single YAML, ideal for feeding into `defectset` / `AutoMLIP`.

```yaml
structure: BN.poscar
supercell: [4, 4, 2]

defects:
  - type: vacancy
    element: N
  - type: vacancy
    element: B
  - type: substitution
    element: N
    substitute: O
  - type: substitution
    element: B
    substitute: C
  - type: antisite
    element: N

combinations:
  singletons: true
  pairs: true
  triples: false
  include_composite: false

calculator:
  name: mace_mp
  model: medium
distortions:
  bond_distortion_min: -0.05
  bond_distortion_max: 0.05
  bond_distortion_steps: 8
  rattle_std: 0.05
  n_rattle: 3
mode: [generate, relax, analyse]
nprocs: 4
```

### Four oxygen vacancies in a perovskite

```yaml
structure: SrTiO3.cif
defects:
  - type: vacancy
    element: O
    count: 4
supercell: [3, 3, 3]
calculator:
  name: mace_mp
  model: large
  device: cuda
distortions:
  bond_distortion_min: -0.15
  bond_distortion_max: 0.15
  bond_distortion_steps: 15
  rattle_std: 0.1
  n_rattle: 5
relaxation:
  fmax: 0.02
  optimizer: FIRE
  max_steps: 1500
mode: [generate, relax, analyse]
```

### Nitrogen vacancy + carbon substitution in hBN

```yaml
structure: BN.poscar
defects:
  - type: vacancy
    element: N
  - type: substitution
    element: B
    substitute: C
supercell: [4, 4, 2]
calculator:
  name: mace_mp
distortions:
  bond_distortion_min: -0.05
  bond_distortion_max: 0.05
  bond_distortion_steps: 10
  rattle_std: 0.05
  n_rattle: 3
mode: [generate, relax, analyse]
nprocs: 4
```

### Antisite defect in GaAs

```yaml
structure: GaAs.vasp
defects:
  - type: antisite
    element: Ga
supercell: [2, 2, 2]
calculator:
  name: mace_mp
distortions:
  bond_distortion_min: -0.1
  bond_distortion_max: 0.1
  bond_distortion_steps: 10
mode: [generate, relax, analyse]
```

### LAMMPS with a Tersoff potential

```yaml
structure: BN.poscar
defects:
  - type: vacancy
    element: N
supercell: [4, 4, 2]
calculator:
  name: lammps
  pair_style: "tersoff"
  pair_coeff: ["* * BNC.tersoff B N"]
distortions:
  bond_distortion_min: -0.05
  bond_distortion_max: 0.05
  bond_distortion_steps: 10
relaxation:
  fmax: 0.01
  max_steps: 2000
mode: [generate, relax, analyse]
nprocs: 8
```

### Only generate structures (no relaxation)

```yaml
structure: BN.poscar
defects:
  - type: vacancy
    element: N
    count: 2
supercell: [4, 4, 2]
distortions:
  bond_distortion_min: -0.05
  bond_distortion_max: 0.05
  bond_distortion_steps: 5
  n_rattle: 2
mode: [generate]
```

Produces POSCAR files in `output_dir/structures/` without running any calculation.

### Fine-grained distortion search

```yaml
distortions:
  bond_distortion_min: -0.02
  bond_distortion_max: 0.05
  bond_distortion_steps: 20
  rattle_std: 0.03
  n_rattle: 5
```

## Outputs

All outputs go to `output_dir/` (default: `defectool_output/`).

- `defectool.db` -- ASE SQLite database with all structures, energies, fmax and metadata.
- `structures/` -- POSCAR files for every generated structure.
- `relaxed_structures.xyz` -- extended XYZ file with all converged structures. Each frame has `REF_energy`, `label`, `defect_type`, `defect_config`, `distortion_type`, `distortion_mag` in the info dict.
- `energy_vs_distortion.pdf` -- energy landscape plot with per-point energy annotations and ground state marker. Only converged structures are plotted.
- `coordination.pdf` -- coordination number distribution and average coordination vs distortion.
- `bond_lengths.pdf` -- bond length distributions comparing pristine, undistorted and ground state.
- `rdf.pdf` -- radial distribution functions.
- `relax_*.log` -- per-structure ASE optimiser logs.

When `combinations:` is enabled and more than one defect configuration is generated, the per-configuration plots get a config-name suffix: `energy_vs_distortion_V_N.pdf`, `rdf_V_B__sub_B_O.pdf`, etc. The summary table is printed once per configuration. The single XYZ file holds frames from all configurations, distinguishable via `info["defect_config"]`.

On the terminal, `analyse` prints a summary table like:

```
Label               Type             Mag         E (eV)    dE (eV)   fmax        Status
----------------------------------------------------------------------------------------
bond_-0.022         bond_distortion -0.022  -1068.387358  -0.0012  0.0089        <-- GS
undistorted         undistorted      0.000  -1068.386100   0.0000  0.0095            ok
bond_+0.011         bond_distortion  0.011  -1068.385901   0.0002  0.0120            ok
rattle_0            rattle           0.050           ---      ---     ---   UNCONVERGED
rattle_2            rattle           0.050           ---      ---     ---   UNCONVERGED
```

Unconverged structures are listed as UNCONVERGED with dashes instead of bogus energies. They are excluded from plots and from ground state selection.

## Working with the database

```python
from ase.db import connect

db = connect("defectool_output/defectool.db")

for row in db.select(is_converged=1):
    print(row.label, row.total_energy, row.final_fmax)
    atoms = row.toatoms()
```

Or from the command line:

```bash
ase db defectool_output/defectool.db -c id,label,status,total_energy,final_fmax
```

## Working with the extended XYZ

```python
from ase.io import read

frames = read("defectool_output/relaxed_structures.xyz", index=":")
for atoms in frames:
    print(atoms.info["label"], atoms.info["REF_energy"])
```

Can also be opened directly in OVITO.
