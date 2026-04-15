#!/bin/bash
# htpolynet -- Example 2 -- Poly(alpha-methylstyrene)
#
# Writes all input files into ./2-polymethylstyrene/ and optionally launches the build.
#
# Usage:
#   bash 2-polymethylstyrene.sh          # set up directory only
#   bash 2-polymethylstyrene.sh --run    # set up and launch htpolynet
#
# Requirements: obabel on PATH, htpolynet installed and on PATH.
#
# Cameron F. Abrams — cfa22@drexel.edu

set -euo pipefail

EXDIR="2-polymethylstyrene"
if [ -d "$EXDIR" ]; then
    echo "Error: directory '$EXDIR' already exists. Remove it first." >&2
    exit 1
fi

mkdir -p "$EXDIR/lib/molecules/inputs" "$EXDIR/lib/molecules/parameterized"
cd "$EXDIR"

# ---------------------------------------------------------------------------
# Generate the alpha-methylstyrene monomer mol2 via obabel.
# The active form is ethylmethylbenzene (EMB).
# Atoms 8 and 9 are renamed C1 and C2 — the reactive vinyl carbons.
# ---------------------------------------------------------------------------
EMB_SMILES="C1=CC(C)=CC=C1CC"
obabel -:"$EMB_SMILES" -ismi --gen2d -opng -O lib/molecules/inputs/EMB.png
obabel -:"$EMB_SMILES" -ismi -omol2 -h --gen3d --title "EMB" \
    | sed s/" 8 C "/" 8 C1"/ \
    | sed s/" 9 C "/" 9 C2"/ \
    | sed s/"UNL1"/"EMB "/ > lib/molecules/inputs/EMB.mol2

# ---------------------------------------------------------------------------
# Configuration file
# ---------------------------------------------------------------------------
cat > pMSTY.yaml << 'YAML'
Title: polystyrene
gromacs:
  gmx: gmx
  gmx_options: -quiet -nobackup
  mdrun: gmx mdrun
  mdrun_options:
    gpu_id: 0
ambertools:
  charge_method: gas
constituents:
  EMB:
    count: 127
densification:
  initial_density: 300.0  # kg/m3
  equilibration:
    - ensemble: min
    - ensemble: nvt
      temperature: 300
      ps: 10
    - ensemble: npt
      temperature: 300
      pressure: 10
      ps: 200
precure:
  preequilibration:
    ensemble: npt
    temperature: 300        # K
    pressure: 1             # bar
    ps: 200
  anneal:
    ncycles: 2
    initial_temperature: 300
    cycle_segments:
      - T: 300
        ps: 0
      - T: 600
        ps: 20
      - T: 600
        ps: 20
      - T: 300
        ps: 20
      - T: 300
        ps: 20
  postequilibration:
    ensemble: npt
    temperature: 300        # K
    pressure: 1             # bar
    ps: 100
CURE:
  controls:
    initial_search_radius: 0.5  # nm
    radial_increment: 0.25      # nm
    max_iterations: 150
    desired_conversion: 0.95
    late_threshhold: 0.85
  drag:
    trigger_distance: 0.6   # nm
    increment: 0.08         # nm
    limit: 0.3              # nm
    equilibration:
      - ensemble: min
      - ensemble: nvt
        temperature: 600
        nsteps: 1000
      - ensemble: npt
        temperature: 600
        pressure: 1
        nsteps: 2000
  relax:
    increment: 0.08         # nm
    equilibration:
      - ensemble: min
      - ensemble: nvt
        temperature: 600
        nsteps: 1000
      - ensemble: npt
        temperature: 600
        pressure: 1
        nsteps: 2000
  equilibrate:
    ensemble: npt
    temperature: 300       # K
    pressure: 1            # bar
    ps: 100
  gromacs:
    rdefault: 0.9          # nm
postcure:
  anneal:
    ncycles: 2
    initial_temperature: 300
    cycle_segments:
      - T: 300
        ps: 0
      - T: 600
        ps: 20
      - T: 600
        ps: 20
      - T: 300
        ps: 20
      - T: 300
        ps: 20
  postequilibration:
    ensemble: npt
    temperature: 300       # K
    pressure: 1            # bar
    ps: 100
reactions:
  - name:        emb1_1
    stage:       cure
    reactants:
      1: EMB
      2: EMB
    product:     EMB~C1-C2~EMB
    probability: 1.0
    atoms:
      A:
        reactant: 1
        resid: 1
        atom: C1
        z: 1
      B:
        reactant: 2
        resid: 1
        atom: C2
        z: 1
    bonds:
      - atoms:
          - A
          - B
        order: 1
  - name:         embCC
    stage:        cap
    reactants:
      1: EMB
    product:      EMBCC
    probability:  1.0
    atoms:
      A:
        reactant: 1
        resid: 1
        atom: C1
        z: 1
      B:
        reactant: 1
        resid: 1
        atom: C2
        z: 1
    bonds:
      - atoms:
          - A
          - B
        order: 2
YAML

# ---------------------------------------------------------------------------
# Optionally launch the build
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--run" ]]; then
    htpolynet run -diag diagnostics.log pMSTY.yaml &> console.log
else
    echo "Setup complete. To run:"
    echo "  cd $EXDIR && htpolynet run -diag diagnostics.log pMSTY.yaml"
fi
