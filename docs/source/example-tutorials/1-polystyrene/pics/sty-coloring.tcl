# Coloring for the polystyrene snapshots.
# Two residue classes:
#   - STY    (mauve / colour 13)  - polymerized styrene units along the chain
#   - STYCC  (orange / colour 3)  - cap-stage-reverted unreacted vinyls
#
# render-snapshot.sh bypasses the auto-generated macros file that
# would otherwise shadow `resname`; plain `resname STY` works in
# both PSF-loaded and autobonded modes.

mol delrep 0 top

mol representation Licorice 0.3 12.0 12.0
mol color ColorID 13
mol selection resname STY
mol addrep top

mol representation Licorice 0.3 12.0 12.0
mol color ColorID 3
mol selection resname STYCC
mol addrep top
