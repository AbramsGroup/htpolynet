#!/bin/bash
# htpolynet -- Example 1 -- Polystyrene
#
# Writes all input files into ./1-polystyrene/ and optionally launches the build.
#
# Usage:
#   bash 1-polystyrene.sh          # set up directory only
#   bash 1-polystyrene.sh --run    # set up and launch htpolynet
#
# Requirements: obabel on PATH, htpolynet installed and on PATH.
#
# Cameron F. Abrams — cfa22@drexel.edu

set -euo pipefail

EXDIR="1-polystyrene"
if [ -d "$EXDIR" ]; then
    echo "Error: directory '$EXDIR' already exists. Remove it first." >&2
    exit 1
fi

mkdir -p "$EXDIR/lib/molecules/inputs" "$EXDIR/lib/molecules/parameterized"
cd "$EXDIR"

# ---------------------------------------------------------------------------
# Generate the styrene monomer mol2 via obabel.
# Atoms 7 and 8 are renamed C1 and C2 — the reactive vinyl carbons.
# ---------------------------------------------------------------------------
STYRENE_SMILES="C1=CC=CC=C1CC"
obabel -:"$STYRENE_SMILES" -ismi --gen2d -opng -O lib/molecules/inputs/STY.png
obabel -:"$STYRENE_SMILES" -ismi -omol2 -h --gen3d --title styrene-active \
    | sed s/" 7 C "/" 7 C1"/ \
    | sed s/" 8 C "/" 8 C2"/ \
    | sed s/"UNL1"/"STY "/ > lib/molecules/inputs/STY.mol2

# ---------------------------------------------------------------------------
# Configuration file
# ---------------------------------------------------------------------------
cat > pSTY.yaml << 'YAML'
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
  STY:
    count: 200
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
  - name:        sty1_1
    stage:       cure
    reactants:
      1: STY
      2: STY
    product:     STY~C1-C2~STY
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
  - name:         styCC
    stage:        cap
    reactants:
      1: STY
    product:      STYCC
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
    htpolynet run -diag diagnostics.log pSTY.yaml
else
    echo "Setup complete. To run:"
    echo "  cd $EXDIR && htpolynet run -diag diagnostics.log pSTY.yaml"
fi
