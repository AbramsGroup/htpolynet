# Coloring overrides for DFA/FDE snapshots.
# Two residue classes:
#   - FDE (furfuryl diepoxide)  --> mauve / colour 13
#   - DFA (difurfuryl diamine)  --> green / colour 7
#
# Plain `resname FDE`/`resname DFA` selectors (render-snapshot.sh
# bypasses the auto-generated macros file that would otherwise
# shadow the `resname` keyword).

mol delrep 0 top

mol representation Licorice 0.3 12.0 12.0
mol color ColorID 13
mol selection resname FDE
mol addrep top

mol representation Licorice 0.3 12.0 12.0
mol color ColorID 7
mol selection resname DFA
mol addrep top

# After cure, opened oxiranes carry -DFEC suffixed residue name for any
# that the cap re-formed; render them subtly off-mauve so they're
# distinguishable from intact FDE.
mol representation Licorice 0.3 12.0 12.0
mol color ColorID 11
mol selection resname FDEC
mol addrep top
