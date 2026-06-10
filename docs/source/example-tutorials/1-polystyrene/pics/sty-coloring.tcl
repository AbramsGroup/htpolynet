# Bulk view for polystyrene snapshots.
# Layered representations:
#   Rep 1  thin Lines on every heavy atom, faded grey.  Gives every
#          styrene monomer its phenyl-and-backbone outline AND lets
#          VMD draw every bond, including the new inter-residue
#          C1-C2 chain bonds formed during cure.
#   Rep 2  STY C1+C2 atoms in thick CPK Licorice (coloured by atom
#          name = element).  These two atoms form the polystyrene
#          backbone; rendering them in one rep means the cure-stage
#          C1↔C2 chain bonds get drawn at full Licorice radius.

mol delrep 0 top

mol representation Lines 1.0
mol color ColorID 6
mol selection "noh"
mol addrep top

mol representation Licorice 0.22 18.0 18.0
mol color Name
mol selection "resname STY and (name C1 or name C2)"
mol addrep top
