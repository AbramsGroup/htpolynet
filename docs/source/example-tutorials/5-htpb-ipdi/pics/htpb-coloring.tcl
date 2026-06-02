# Coloring overrides for HTPB-IPDI snapshots.
# Two residue classes (HTPB chain residues bundled, IPD called out
# separately):
#   - HTPB chain residues (OB / TB / TBO)  --> mauve / colour 13
#   - IPD (the urethane crosslinker)       --> green / colour 7

mol delrep 0 top

mol representation Licorice 0.25 12.0 12.0
mol color ColorID 13
mol selection "resname OB or resname TB or resname TBO"
mol addrep top

mol representation Licorice 0.3 12.0 12.0
mol color ColorID 7
mol selection resname IPD
mol addrep top
