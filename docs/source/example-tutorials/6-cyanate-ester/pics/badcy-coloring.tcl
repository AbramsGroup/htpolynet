# Coloring overrides for BADCy snapshots.
# Three residue classes:
#   - BPA  (mauve / colour 13)  - the bisphenol-A bridges
#   - TAZ  (green / colour 7)   - intact 1,3,5-triazine crosslinks
#   - CYN  (red / colour 1)     - post-repair -C#N end-groups
#
# render-snapshot.sh deliberately does NOT source the
# .viz.macros.tcl that htpolynet emits alongside the PSF, because the
# auto-generated macros (named after residue names) shadow VMD's
# built-in `resname` keyword.  Plain `resname BPA` selectors work in
# both PSF-loaded and autobonded modes.

mol delrep 0 top

mol representation Licorice 0.3 12.0 12.0
mol color ColorID 13
mol selection resname BPA
mol addrep top

mol representation Licorice 0.3 12.0 12.0
mol color ColorID 7
mol selection resname TAZ
mol addrep top

mol representation Licorice 0.3 12.0 12.0
mol color ColorID 1
mol selection resname CYN
mol addrep top
