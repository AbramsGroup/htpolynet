#!/bin/bash
# htpolynet -- Example 4 -- DFDA/FDE Furan-based Epoxy Thermoset
#
# Writes all input files into ./4-dfda-fde-epoxy-thermoset/ and optionally launches the build.
#
# Usage:
#   bash 4-dfda-fde-epoxy-thermoset.sh          # set up directory only
#   bash 4-dfda-fde-epoxy-thermoset.sh --run    # set up and launch htpolynet
#
# Requirements: obabel on PATH, htpolynet installed and on PATH.
#
# Note: obabel outputs PDB format for these monomers (mol2 atom-numbering is
# less predictable for fused heterocyclic systems).
#
# Cameron F. Abrams — cfa22@drexel.edu

set -euo pipefail

EXDIR="4-dfda-fde-epoxy-thermoset"
if [ -d "$EXDIR" ]; then
    echo "Error: directory '$EXDIR' already exists. Remove it first." >&2
    exit 1
fi

mkdir -p "$EXDIR/lib/molecules/inputs" "$EXDIR/lib/molecules/parameterized"
cd "$EXDIR"

# ---------------------------------------------------------------------------
# Generate monomer PDB files via obabel.
# ---------------------------------------------------------------------------

FOURAMINOFURANYL="C1OC(CN)=CC=1"
FURANYL="c1occc1"

# DFDA (difurfuryl diamine)
# Reactive atoms: N1 (atom 6), N2 (atom 13)
DFDA_SMILES="C(${FOURAMINOFURANYL})${FOURAMINOFURANYL}"
obabel -:"$DFDA_SMILES" -ismi --gen2d -opng -O lib/molecules/inputs/DFA.png
obabel -:"$DFDA_SMILES" -ismi -h --gen3d --title DFA -opdb \
    | sed s/"UNL "/"DFA "/ \
    | sed s/"HETATM    6  N "/"HETATM    6  N1"/ \
    | sed s/"HETATM   13  N "/"HETATM   13  N2"/ \
    > lib/molecules/inputs/DFA.pdb

# FDE (furfuryl diepoxide)
# Reactive atoms: C1 (atom 11), C2 (atom 15), C3 (atom 9), C4 (atom 13)
FDE_SMILES="N((C${FURANYL})CC(O)C)CC(O)C"
obabel -:"$FDE_SMILES" -ismi --gen2d -opng -O lib/molecules/inputs/FDE.png
obabel -:"$FDE_SMILES" -ismi -h --gen3d --title FDE -opdb \
    | sed s/"UNL "/"FDE "/ \
    | sed s/"HETATM   11  C "/"HETATM   11  C1"/ \
    | sed s/"HETATM   15  C "/"HETATM   15  C2"/ \
    | sed s/"HETATM    9  C "/"HETATM    9  C3"/ \
    | sed s/"HETATM   13  C "/"HETATM   13  C4"/ \
    > lib/molecules/inputs/FDE.pdb

# ---------------------------------------------------------------------------
# Configuration file
# ---------------------------------------------------------------------------
cat > FDEDFA.yaml << 'YAML'
Title: DFA-FDE thermoset
gromacs: {
  gmx: 'gmx',
  gmx_options: '-quiet -nobackup',
  mdrun: 'gmx mdrun',
  mdrun_options: {'gpu_id': 0}
}
ambertools: {
  charge_method: gas
}
constituents: {
  FDE: {count: 200, symmetry_equivalent_atoms: [[C1,C2],[C3,C4],[O1,O2]], stereocenters: [C3], nconformers: 2},
  DFA: {count: 100, symmetry_equivalent_atoms: [[N1,N2]], nconformers: 4}
}
densification: {
  initial_density: 300.0,  # kg/m3
  equilibration: [
    { ensemble: min },
    { ensemble: nvt, temperature: 300, ps: 10 },
    { ensemble: npt, temperature: 300, pressure: 10, ps: 300 }
  ]
}
precure: {
  preequilibration: {
    ensemble: npt,
    temperature: 300,        # K
    pressure: 1,             # bar
    ps: 200
  },
  anneal: {
    ncycles: 2,
    initial_temperature: 300,
    cycle_segments: [
      { T: 300, ps: 0 },
      { T: 600, ps: 20 },
      { T: 600, ps: 20 },
      { T: 300, ps: 20 },
      { T: 300, ps: 20 }
    ]
  },
  postequilibration: {
    ensemble: npt,
    temperature: 300,        # K
    pressure: 1,             # bar
    ps: 100
  }
}
CURE: {
  controls: {
    initial_search_radius: 0.5,  # nm
    radial_increment: 0.25,      # nm
    max_iterations: 150,
    desired_conversion: 0.95,
    late_threshhold: 0.85
  },
  drag: {
    trigger_distance: 0.6,   # nm
    increment: 0.08,         # nm
    limit: 0.3,              # nm
    equilibration: [
      { ensemble: min },
      { ensemble: nvt, temperature: 600, nsteps: 1000 },
      { ensemble: npt, temperature: 600, pressure: 1, nsteps: 2000 }
    ]
  },
  relax: {
    increment: 0.08,         # nm
    equilibration: [
      { ensemble: min },
      { ensemble: nvt, temperature: 600, nsteps: 1000 },
      { ensemble: npt, temperature: 600, pressure: 1, nsteps: 2000 }
    ]
  },
  equilibrate: {
    ensemble: npt,
    temperature: 300,        # K
    pressure: 1,             # bar
    ps: 100
  },
  gromacs: {
    rdefault: 0.9            # nm
  }
}
postcure: {
  anneal: {
    ncycles: 2,
    initial_temperature: 300,
    cycle_segments: [
      { T: 300, ps: 0 },
      { T: 600, ps: 20 },
      { T: 600, ps: 20 },
      { T: 300, ps: 20 },
      { T: 300, ps: 20 }
    ]
  },
  postequilibration: {
    ensemble: npt,
    temperature: 300,        # K
    pressure: 1,             # bar
    ps: 100
  }
}
reactions:
  - {
      name:        'Primary-to-secondary-amine',
      stage:       cure,
      reactants:   {1: DFA, 2: FDE},
      product:     DFA~N1-C1~FDE,
      probability: 1.0,
      atoms: {
        A: {reactant: 1, resid: 1, atom: N1, z: 2},
        B: {reactant: 2, resid: 1, atom: C1, z: 1}
      },
      bonds: [
        {atoms: [A, B], order: 1}
      ]
    }
  - {
      name:        'Secondary-to-tertiary-amine',
      reactants:   {1: DFA~N1-C1~FDE, 2: FDE},
      product:     DFA~N1-C1~FDE-C1~FDE,
      stage:       cure,
      probability: 0.5,
      atoms: {
        A: {reactant: 1, resid: 1, atom: N1, z: 1},
        B: {reactant: 2, resid: 1, atom: C1, z: 1}
      },
      bonds: [
        {atoms: [A, B], order: 1}
      ]
    }
  - {
      name:        'Oxirane-formation',
      reactants:   {1: FDE},
      product:     FDEC,
      stage:       cap,
      probability: 1.0,
      atoms: {
        A: {reactant: 1, resid: 1, atom: O1, z: 1},
        B: {reactant: 1, resid: 1, atom: C1, z: 1}
      },
      bonds: [
        {atoms: [A, B], order: 1}
      ]
    }
YAML

# ---------------------------------------------------------------------------
# Optionally launch the build
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--run" ]]; then
    htpolynet run -diag diagnostics.log FDEDFA.yaml &> console.log
else
    echo "Setup complete. To run:"
    echo "  cd $EXDIR && htpolynet run -diag diagnostics.log FDEDFA.yaml"
fi
