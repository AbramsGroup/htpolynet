# Render a detail snapshot of one complete crosslink site.
#
# Driven by environment variables — see render-detail.sh for the
# shell wrapper that exports them.  Env vars:
#   PSF_PATH, GRO_PATH       - input system
#   SNAPSHOT_OUT, TACHYON    - output TGA path and tachyon binary
#   NODE_RESNAME             - the crosslink-node residue (TAZ, PAC, ...)
#   PARTNER_RESNAMES         - whitespace-separated list of partner residues
#                              (BPA, DGE, ...) to follow via the bond graph
#   FACE_ON                  - "1" to align the node ring normal with +z
#   FACE_ATOMS               - atom-name pattern for the ring plane atoms
#                              (used only when FACE_ON=1, e.g. "N1 N2 N3")
#
# Algorithm:
#   1. Load PSF + GRO; box dims come from the gro's last line (nm → Å).
#   2. Pick the NODE_RESNAME residue whose centroid is closest to the
#      box center (avoids straddling a periodic boundary).
#   3. Follow the bond graph from every atom of that residue, collecting
#      every PARTNER_RESNAME residue that's directly bonded.
#   4. BFS-unwrap the subset using the bond graph — each newly-visited
#      atom gets shifted by integer box vectors so it sits within box/2
#      of its bonded predecessor.  Handles BOTH residue-internal PBC
#      wraps AND inter-residue PBC wraps in one pass.
#   5. Trim residual long PBC-crossing bonds.
#   6. Write the subset (node + partners) to a PDB, reload it with
#      autobonds, and render with a single thick CPK Licorice rep.
#   7. Optionally rotate so the node ring normal aligns with +z.

package require topotools
package require pbctools

set PSF $env(PSF_PATH)
set GRO $env(GRO_PATH)
set OUT $env(SNAPSHOT_OUT)
set TACH $env(TACHYON)
set NODE_RESNAME $env(NODE_RESNAME)
set PARTNER_RESNAMES $env(PARTNER_RESNAMES)
set FACE_ON [expr {[info exists env(FACE_ON)] && $env(FACE_ON) == "1"}]
if {$FACE_ON} { set FACE_ATOMS $env(FACE_ATOMS) }
set SUBSET /tmp/render-detail-subset.pdb

mol new $PSF type psf waitfor all
mol addfile $GRO type gro waitfor all

# Box dims from the gro's last line (nm → Å)
set last_line [exec tail -1 $GRO]
set bx [expr {[lindex $last_line 0] * 10.0}]
set by [expr {[lindex $last_line 1] * 10.0}]
set bz [expr {[lindex $last_line 2] * 10.0}]
set cx [expr {$bx / 2.0}]
set cy [expr {$by / 2.0}]
set cz [expr {$bz / 2.0}]

# Center-most node residue
set node_residues [lsort -integer -unique [[atomselect top "resname $NODE_RESNAME"] get residue]]
set best_node -1
set best_d2 1e9
foreach r $node_residues {
    set sel [atomselect top "residue $r"]
    set c [measure center $sel]
    $sel delete
    set dx [expr {[lindex $c 0] - $cx}]
    set dy [expr {[lindex $c 1] - $cy}]
    set dz [expr {[lindex $c 2] - $cz}]
    set d2 [expr {$dx*$dx + $dy*$dy + $dz*$dz}]
    if {$d2 < $best_d2} { set best_d2 $d2; set best_node $r }
}
puts "Center-most $NODE_RESNAME residue: $best_node  (d=[expr {sqrt($best_d2)}] Å)"

# Walk the bond graph from node atoms to find bonded partner residues.
set node_atoms [atomselect top "residue $best_node"]
set partner_set [list]
foreach atom_bonds [$node_atoms getbonds] {
    foreach partner $atom_bonds {
        set p [atomselect top "index $partner"]
        set p_res [$p get resname]
        set p_id [$p get residue]
        $p delete
        if {[lsearch -exact $PARTNER_RESNAMES $p_res] >= 0 && $p_id != $best_node} {
            lappend partner_set $p_id
        }
    }
}
$node_atoms delete
set partner_set [lsort -integer -unique $partner_set]
puts "Bonded partner residues: $partner_set"

# Build the subset selection string
set sel_str "residue $best_node"
foreach r $partner_set { append sel_str " or residue $r" }

# BFS-unwrap the subset.  Starting from one atom of the node, walk
# the bond graph (intra- and inter-residue) and shift each new atom
# by integer box vectors so it sits within box/2 of its bonded
# predecessor.  Handles both residue-internal PBC wraps and inter-
# residue PBC wraps in one pass.
set node_sel [atomselect top "residue $best_node"]
set node_indices [$node_sel list]
$node_sel delete
set anchor [lindex $node_indices 0]

set subset_sel [atomselect top $sel_str]
array set in_subset {}
foreach i [$subset_sel list] { set in_subset($i) 1 }
$subset_sel delete

array set placed {}
set placed($anchor) 1
set queue [list $anchor]
set nshifts 0
while {[llength $queue] > 0} {
    set current [lindex $queue 0]
    set queue [lrange $queue 1 end]
    set csel [atomselect top "index $current"]
    set cpos [lindex [$csel get {x y z}] 0]
    set cbonds [lindex [$csel getbonds] 0]
    $csel delete
    foreach next $cbonds {
        if {![info exists in_subset($next)]} continue
        if {[info exists placed($next)]} continue
        set nsel [atomselect top "index $next"]
        set npos [lindex [$nsel get {x y z}] 0]
        set dx [expr {[lindex $npos 0] - [lindex $cpos 0]}]
        set dy [expr {[lindex $npos 1] - [lindex $cpos 1]}]
        set dz [expr {[lindex $npos 2] - [lindex $cpos 2]}]
        set sx [expr {round($dx / $bx)}]
        set sy [expr {round($dy / $by)}]
        set sz [expr {round($dz / $bz)}]
        if {$sx != 0 || $sy != 0 || $sz != 0} {
            $nsel moveby [list [expr {-$sx * $bx}] [expr {-$sy * $by}] [expr {-$sz * $bz}]]
            incr nshifts
        }
        $nsel delete
        set placed($next) 1
        lappend queue $next
    }
}
puts "BFS unwrap: shifted $nshifts atoms"

# Trim residual long bonds (PBC crossings)
set _bl [topo getbondlist both]
set _keep [list]
foreach _b $_bl {
    lassign $_b _i _j _t _o
    if {[measure bond [list $_i $_j]] <= 3.0} { lappend _keep $_b }
}
topo setbondlist both $_keep

# Write the subset, then reload it as a fresh molecule for rendering
set local [atomselect top $sel_str]
puts "Subset atoms: [$local num]"
$local writepdb $SUBSET
$local delete

mol delete top
mol new $SUBSET type pdb autobonds yes waitfor all

# Single thick CPK Licorice rep on heavy atoms — colours each atom
# by its element via the first character of its name (C grey, N blue,
# O red, etc.).  Putting every atom in one rep means every bond gets
# drawn at full Licorice radius, including the inter-residue cure
# bonds.
mol delrep 0 top
mol representation Licorice 0.3 25.0 25.0
mol color Name
mol selection "noh"
mol addrep top

display projection Orthographic
display nearclip set 0.01
display depthcue off
color Display Background white

display resetview

# Optional face-on rotation: align the node ring normal with +z so
# the ring renders face-on instead of edge-on.
if {$FACE_ON} {
    set ring_sel [atomselect top "resname $NODE_RESNAME and ($FACE_ATOMS)"]
    set positions [$ring_sel get {x y z}]
    $ring_sel delete
    set p1 [lindex $positions 0]
    set p2 [lindex $positions 1]
    set p3 [lindex $positions 2]
    set v1 [vecsub $p2 $p1]
    set v2 [vecsub $p3 $p1]
    set normal [vecnorm [veccross $v1 $v2]]
    set zaxis {0.0 0.0 1.0}
    set cos_a [vecdot $normal $zaxis]
    if {$cos_a > 0.99999} {
        puts "Ring normal already along +z"
    } elseif {$cos_a < -0.99999} {
        molinfo top set rotate_matrix [list [transabout {1.0 0.0 0.0} 180 deg]]
        puts "Ring normal anti-parallel to z; rotated 180° around x"
    } else {
        set axis [vecnorm [veccross $normal $zaxis]]
        set angle [expr {acos($cos_a) * 180.0 / 3.141592653589793}]
        molinfo top set rotate_matrix [list [transabout $axis $angle deg]]
        puts "Rotated by $angle deg around $axis to align ring normal with +z"
    }
} else {
    rotate x by -30
    rotate y by 30
}

render Tachyon $OUT "$TACH" -aasamples 12 %s -format TARGA -res 1200 900 -o %s
quit
