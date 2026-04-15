#!/bin/bash
# htpolynet -- Example 3 -- Bis-GMA/Styrene Thermoset
#
# Writes all input files into ./3-bisgma-styrene-thermoset/ and optionally launches the build.
#
# Usage:
#   bash 3-bisgma-styrene-thermoset.sh          # set up directory only
#   bash 3-bisgma-styrene-thermoset.sh --run    # set up and launch htpolynet
#
# Requirements: obabel on PATH, htpolynet installed and on PATH.
#
# The bis-GMA monomer (GMA) is assembled from bisphenol-A (BPA) and
# 1-hydroxyethyl methacrylate (HIE) via two param-stage reactions specified
# in the config file.
#
# Cameron F. Abrams — cfa22@drexel.edu

set -euo pipefail

EXDIR="3-bisgma-styrene-thermoset"
if [ -d "$EXDIR" ]; then
    echo "Error: directory '$EXDIR' already exists. Remove it first." >&2
    exit 1
fi

mkdir -p "$EXDIR/lib/molecules/inputs" "$EXDIR/lib/molecules/parameterized"
cd "$EXDIR"

# ---------------------------------------------------------------------------
# Generate monomer mol2 files via obabel.
# ---------------------------------------------------------------------------

# Styrene (active form) — reactive vinyl carbons C1 (atom 7) and C2 (atom 8)
STYRENE="C1=CC=CC=C1CC"
obabel -:"$STYRENE" -ismi --gen2d -opng -O lib/molecules/inputs/STY.png
obabel -:"$STYRENE" -ismi -omol2 -h --gen3d --title "Styrene-active" \
    | sed s/" 7 C "/" 7 C1"/ \
    | sed s/" 8 C "/" 8 C2"/ \
    | sed s/"UNL1"/"STY "/ > lib/molecules/inputs/STY.mol2

# Bisphenol-A — reactive oxygens O1 (atom 7) and O2 (atom 14)
PHENOL="C1=CC=C(O)C=C1"
BPA="CC($PHENOL)($PHENOL)C"
obabel -:"$BPA" -ismi --gen2d -opng -O lib/molecules/inputs/BPA.png -xp 600
obabel -:"$BPA" -ismi --gen3d -h -omol2 --title "Bisphenol A" \
    | sed s/"UNL1"/"BPA "/ \
    | sed s/" 7 O "/" 7 O1"/ \
    | sed s/"14 O "/"14 O2"/ \
    > lib/molecules/inputs/BPA.mol2

# 1-hydroxyethyl methacrylate (HIE) — reactive carbons C1 (atom 2), C2 (atom 10), C3 (atom 7), C4 (atom 9)
HIE="CC(C(=O)OCC(O)C)C"
obabel -:"$HIE" -ismi --gen2d -opng -O lib/molecules/inputs/HIE.png -xp 600
obabel -:"$HIE" -ismi --gen3d -h -omol2 --title "1-hydroxyethyl methacrylate" \
    | sed s/"UNL1"/"HIE "/ \
    | sed s/"10 C "/"10 C2"/ \
    | sed s/" 2 C "/" 2 C1"/ \
    | sed s/" 7 C "/" 7 C3"/ \
    | sed s/" 9 C "/" 9 C4"/ \
    > lib/molecules/inputs/HIE.mol2

# ---------------------------------------------------------------------------
# Configuration file
# ---------------------------------------------------------------------------
cat > GMASTY.yaml << 'YAML'
Title: GMA-STY thermoset
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
    count: 150
  GMA:
    count: 75
  HIE:
    stereocenters:
      - C1
      - C3
densification:
  initial_density: 100.0  # kg/m3
  equilibration:
    - ensemble: min
    - ensemble: nvt
      temperature: 300
      ps: 10
    - ensemble: npt
      temperature: 300
      pressure: 10
      ps: 100
      repeat: 8
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
  # Build bis-GMA by adding one hydroxyethyl methacrylate to each side of BPA
  - name: B1
    stage: param
    reactants:
      1: BPA
      2: HIE
    product: GM1
    atoms:
      A:
        reactant: 1
        resid: 1
        atom: O1
        z: 1
      B:
        reactant: 2
        resid: 1
        atom: C4
        z: 2
    bonds:
      - atoms: [A, B]
        order: 1
  - name: B2
    stage: param
    reactants:
      1: GM1
      2: HIE
    product: GMA
    atoms:
      A:
        reactant: 1
        resid: 1
        atom: O2
        z: 1
      B:
        reactant: 2
        resid: 1
        atom: C4
        z: 1
    bonds:
      - atoms: [A, B]
        order: 1
  # Cure-stage polymerization reactions
  - name: dimer_xx
    stage: cure
    reactants:
      1: STY
      2: STY
    product: STY~C1-C2~STY
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
      - atoms: [A, B]
        order: 1
  - name: dimer_yy
    stage: cure
    reactants:
      1: HIE
      2: HIE
    product: HIE~C1-C2~HIE
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
      - atoms: [A, B]
        order: 1
  - name: dimer_xy
    stage: cure
    reactants:
      1: STY
      2: HIE
    product: STY~C1-C2~HIE
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
      - atoms: [A, B]
        order: 1
  - name: dimer_yx
    stage: cure
    reactants:
      1: HIE
      2: STY
    product: HIE~C1-C2~STY
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
      - atoms: [A, B]
        order: 1
  # Capping reactions
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
      - atoms: [A, B]
        order: 2
  - name:         hieCC
    stage:        cap
    reactants:
      1: HIE
    product:      HIECC
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
      - atoms: [A, B]
        order: 2
YAML

# ---------------------------------------------------------------------------
# Optionally launch the build
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--run" ]]; then
    htpolynet run -diag diagnostics.log GMASTY.yaml &> console.log
else
    echo "Setup complete. To run:"
    echo "  cd $EXDIR && htpolynet run -diag diagnostics.log GMASTY.yaml"
fi
