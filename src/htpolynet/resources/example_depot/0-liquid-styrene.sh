#!/bin/bash
# htpolynet -- Example 0 -- Liquid Styrene
#
# Writes all input files into ./0-liquid-styrene/ and optionally launches the build.
#
# Usage:
#   bash 0-liquid-styrene.sh          # set up directory only
#   bash 0-liquid-styrene.sh --run    # set up and launch htpolynet
#
# Requirements: obabel on PATH, htpolynet installed and on PATH.
#
# After the build, run the post-simulation analyses with:
#   cd 0-liquid-styrene
#   htpolynet analyze -cfg ck.yaml -ocfg STY.yaml -proj proj-0
#
# Cameron F. Abrams — cfa22@drexel.edu

set -euo pipefail

EXDIR="0-liquid-styrene"
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
# Main configuration file
# ---------------------------------------------------------------------------
cat > STY.yaml << 'YAML'
Title: Liquid of Active Styrene Monomers
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
      temperature: 300.0
      ps: 10.0
    - ensemble: npt
      temperature: 300.0
      pressure: 10.0
      ps: 200.0
precure:
  preequilibration:
    ensemble: npt
    temperature: 300        # K
    pressure: 1.0           # bar
    ps: 200
  anneal:
    ncycles: 2
    initial_temperature: 300.0
    cycle_segments:
      - T: 300.0
        ps: 0
      - T: 600.0
        ps: 20
      - T: 600.0
        ps: 20
      - T: 300.0
        ps: 20
      - T: 300.0
        ps: 20
  postequilibration:
    ensemble: npt
    temperature: 300.0      # K
    pressure: 1.0           # bar
    ps: 100
YAML

# ---------------------------------------------------------------------------
# Post-simulation analysis configuration file
# ---------------------------------------------------------------------------
cat > ck.yaml << 'YAML'
- command: check
  subdir: analyze/check
  links:
    - systems/densification/densified-npt.trr
  options:
    f: densified-npt.trr
  outfile: check.out
  matchlines: ['Last frame','Coords']
- command: report-methods
  subdir: analyze/report-methods
  links:
    - systems/densification/densified-npt.tpr
  options:
    s: densified-npt.tpr
    m: report.tex
- command: freevolume
  subdir: analyze/freevolume
  links:
    - systems/densification/densified-npt.tpr
    - systems/densification/densified-npt.trr
  options:
    s: densified-npt.tpr
    f: densified-npt.trr
YAML

# ---------------------------------------------------------------------------
# Optionally launch the build
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--run" ]]; then
    htpolynet run -diag diagnostics.log STY.yaml
else
    echo "Setup complete. To run:"
    echo "  cd $EXDIR && htpolynet run -diag diagnostics.log STY.yaml"
    echo ""
    echo "After the build, to run post-simulation analyses:"
    echo "  cd $EXDIR && htpolynet analyze -cfg ck.yaml -ocfg STY.yaml -proj proj-0"
fi
