#!/bin/bash
# htpolynet -- Example 4 -- PACM/DGEBA Epoxy Thermoset
#
# Writes all input files into ./4-pacm-dgeba-epoxy-thermoset/ and optionally launches the build.
#
# Usage:
#   bash 4-pacm-dgeba-epoxy-thermoset.sh          # set up directory only
#   bash 4-pacm-dgeba-epoxy-thermoset.sh --run    # set up and launch htpolynet
#
# Requirements: obabel on PATH, htpolynet installed and on PATH.
#
# Cameron F. Abrams — cfa22@drexel.edu

set -euo pipefail

EXDIR="4-pacm-dgeba-epoxy-thermoset"
if [ -d "$EXDIR" ]; then
    echo "Error: directory '$EXDIR' already exists. Remove it first." >&2
    exit 1
fi

mkdir -p "$EXDIR/lib/molecules/inputs" "$EXDIR/lib/molecules/parameterized"
cd "$EXDIR"

# ---------------------------------------------------------------------------
# Generate monomer mol2 files via obabel.
# ---------------------------------------------------------------------------

# DGEBA (diglycidyl ether of bisphenol-A)
# Reactive atoms: C1/C2 (epoxide carbons, atoms 14/25), C3/C4 (atoms 12/23), O1/O2 (epoxide oxygens, atoms 13/24)
DGEBA="CC(C)(C1=CC=C(C=C1)OCC(O)C)C3=CC=C(C=C3)OCC(O)C"
obabel -:"$DGEBA" -ismi --gen2d -opng -O lib/molecules/inputs/DGE.png
obabel -:"$DGEBA" -ismi -h --gen3d -omol2 --title "DGE" \
    | sed s/"UNL1   "/"DGE    "/ \
    | sed s/"14 C "/"14 C1"/ \
    | sed s/"25 C "/"25 C2"/ \
    | sed s/"12 C "/"12 C3"/ \
    | sed s/"23 C "/"23 C4"/ \
    | sed s/"13 O "/"13 O1"/ \
    | sed s/"24 O "/"24 O2"/ > lib/molecules/inputs/DGE.mol2

# PACM (4,4'-methylenebis(cyclohexylamine))
# Reactive atoms: N1/N2 (amine nitrogens, atoms 14/15), C1/C2 (atoms 3/11)
PACM="C1CC(CCC1CC2CCC(CC2)N)N"
obabel -:"$PACM" -ismi --gen2d -opng -O lib/molecules/inputs/PAC.png
obabel -:"$PACM" -ismi -h --gen3d -omol2 --title "PAC" \
    | sed s/"UNL1   "/"PAC    "/ \
    | sed s/"14 N "/"14 N1"/ \
    | sed s/"15 N "/"15 N1"/ \
    | sed s/"3 C "/"3 C1"/ \
    | sed s/"11 C "/"11 C1"/ > lib/molecules/inputs/PAC.mol2

# ---------------------------------------------------------------------------
# Configuration file
# ---------------------------------------------------------------------------
cat > DGEPAC.yaml << 'YAML'
Title: DGEBA-PACM thermoset
gromacs:
  gmx: gmx
  gmx_options: -quiet -nobackup
  mdrun: gmx mdrun
  mdrun_options:
    gpu_id: 0
ambertools:
  charge_method: gas
constituents:
  DGE:
    count: 200
    symmetry_equivalent_atoms: [[C1,C2],[C3,C4],[O1,O2]]
    stereocenters: [C3]
  PAC:
    count: 100
    symmetry_equivalent_atoms: [[N1,N2],[C1,C2]]
    stereocenters: [C1]
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
      ps: 300
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
  - name:        'Primary-to-secondary-amine'
    stage:       cure
    reactants:
      1: PAC
      2: DGE
    product:     PAC~N1-C1~DGE
    probability: 1.0
    atoms:
      A:
        reactant: 1
        resid: 1
        atom: N1
        z: 2
      B:
        reactant: 2
        resid: 1
        atom: C1
        z: 1
    bonds:
      - atoms: [A, B]
        order: 1
  - name:        'Secondary-to-tertiary-amine'
    reactants:
      1: PAC~N1-C1~DGE
      2: DGE
    product:     PAC~N1-C1~DGE-C1~DGE
    stage:       cure
    probability: 0.5
    atoms:
      A:
        reactant: 1
        resid: 1
        atom: N1
        z: 1
      B:
        reactant: 2
        resid: 1
        atom: C1
        z: 1
    bonds:
      - atoms: [A, B]
        order: 1
  - name:        'Oxirane-formation'
    reactants:
      1: DGE
    product:     DGEC
    stage:       cap
    probability: 1.0
    atoms:
      A:
        reactant: 1
        resid: 1
        atom: O1
        z: 1
      B:
        reactant: 1
        resid: 1
        atom: C1
        z: 1
    bonds:
      - atoms: [A, B]
        order: 1
YAML

# ---------------------------------------------------------------------------
# Optionally launch the build
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--run" ]]; then
    htpolynet run -diag diagnostics.log DGEPAC.yaml &> console.log
else
    echo "Setup complete. To run:"
    echo "  cd $EXDIR && htpolynet run -diag diagnostics.log DGEPAC.yaml"
fi
