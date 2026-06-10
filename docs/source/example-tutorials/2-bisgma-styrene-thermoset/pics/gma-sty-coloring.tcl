# Bulk view for BisGMA / styrene snapshots.
# Layered representations:
#   Rep 1  thin Lines on every heavy atom, faded grey.  Gives the GMA
#          + STY bulk its monomer outline AND lets VMD draw every
#          bond, including the new inter-residue C1-C2 chain bonds.
#   Rep 2  STY and HIE chain-carbons (C1, C2) in thick CPK Licorice
#          (coloured by atom name = element).  These are the vinyl
#          carbons that polymerize during cure; rendering them in a
#          single rep means the new C1↔C2 chain bonds (whether
#          STY↔STY, HIE↔STY, or HIE↔HIE) draw at full Licorice
#          radius.

mol delrep 0 top

mol representation Lines 1.0
mol color ColorID 6
mol selection "noh"
mol addrep top

mol representation Licorice 0.22 18.0 18.0
mol color Name
mol selection "(resname STY or resname HIE) and (name C1 or name C2)"
mol addrep top
