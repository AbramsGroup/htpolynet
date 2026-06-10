# Bulk view for HTPB / IPDI snapshots.
# The HTPB butadiene matrix (TB residues) is hidden — ex 5's
# crosslink density is so low (~1 IPDI per 100 butadiene units)
# that drawing the matrix swamps the urethane sites visually.
# Instead, we render only the IPDI crosslinkers (IPD residue) and
# the HTPB chain end-groups (TBO residue) in CPK Licorice.
#
# In the liquid: IPDIs and HTPB end-groups float as disconnected
# pieces.  In the cured: each IPDI tethers to 2 TBOs through the
# urethane bond, so the visible "stars" each get 2 TBO arms.

mol delrep 0 top

mol representation Licorice 0.25 25.0 25.0
mol color Name
mol selection "(resname IPD or resname TBO) and noh"
mol addrep top
