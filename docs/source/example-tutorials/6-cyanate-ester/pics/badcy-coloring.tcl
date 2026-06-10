# Bulk view for BADCy snapshots.
# Layered representations:
#   Rep 1  thin Lines on every heavy atom, faded grey.  Gives the
#          BPA backbones their phenyl-aryl outlines AND lets VMD draw
#          every bond, including the new inter-residue cure bonds.
#   Rep 2  BPA phenolic O atoms (O1, O2), all TAZ ring atoms, and
#          all CYN cap atoms in thick CPK Licorice (coloured by atom
#          name = element).  Putting every chemistry-relevant atom
#          into a single rep means the inter-residue cure bonds
#          (BPA-O → TAZ-C aryl-ether and BPA-O → CYN-C cap bonds)
#          draw at full Licorice radius.

mol delrep 0 top

mol representation Lines 1.0
mol color ColorID 6
mol selection "noh"
mol addrep top

mol representation Licorice 0.22 18.0 18.0
mol color Name
mol selection "((resname BPA and (name O1 or name O2)) or resname TAZ or resname CYN) and noh"
mol addrep top
