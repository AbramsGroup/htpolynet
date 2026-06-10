# Bulk view for DFA / FDE snapshots.
# Layered representations:
#   Rep 1  thin Lines on every heavy atom, faded grey.  Gives DFA
#          (difurfuryl diamine) and FDE (furfuryl diepoxide) their
#          molecular outlines AND lets VMD draw every bond, including
#          the new inter-residue N-C crosslink bonds formed during
#          cure.
#   Rep 2  DFA amine nitrogens (N1, N2) + the adjacent ring carbons,
#          plus FDE epoxide carbons (C1, C2), the cure-opened
#          hydroxyl carbons (C3, C4), and all FDE O atoms in thick
#          CPK Licorice (coloured by atom name = element).  This
#          captures the full amine→epoxide chemistry region per
#          crosslink — N atoms (blue) and O atoms (red) make the
#          crosslink sites visually distinct against the grey carbon
#          backbone.

mol delrep 0 top

mol representation Lines 1.0
mol color ColorID 6
mol selection "noh"
mol addrep top

mol representation Licorice 0.22 18.0 18.0
mol color Name
mol selection "(resname DFA and (name N1 or name N2 or name C1 or name C2)) or (resname FDE and (name C1 or name C2 or name C3 or name C4 or name O or name O1 or name O2))"
mol addrep top
