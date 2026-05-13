"""Write VMD-friendly viz files (PSF + TCL) from a gromacs top + gro pair.

The PSF carries the real bond topology so VMD does not have to guess bonds
from interatomic distances; the TCL drops bonds longer than a cutoff from
the display so periodic-boundary-crossing bonds in a crosslinked network do
not render as long sticks.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import logging
import os

logger = logging.getLogger(__name__)


_TCL_TEMPLATE = '''\
# htpolynet VMD viz helper.
# Run with:  vmd {psf} {gro} -e {tcl}
# Removes bonds longer than the cutoff (Å) from display so PBC-crossing
# bonds in a crosslinked network do not draw as "long bonds".  The
# underlying topology (atoms, bonds in the .psf) is unchanged; only the
# per-frame bond list shown by VMD is filtered.
#
# Wrapped in a proc so top-level return values (which VMD echoes to the
# console) stay hidden — only the explicit `puts` lines below are seen.

package require topotools

proc htpolynet_trim_pbc_bonds {{{{cutoff 3.0}}}} {{
    set bl [topo getbondlist both]
    if {{[llength $bl] == 0}} {{
        puts "viz: PSF appears to carry no bonds — nothing to filter."
        return
    }}
    # Sanity check: a typical covalent bond should report ~1–2 Å.  If it
    # reports ~0.1–0.2, VMD loaded coordinates in nm — raise the cutoff.
    lassign [lindex $bl 0] i0 j0 _t _o
    set d0 [measure bond [list $i0 $j0]]
    puts [format "viz: first bond %d-%d = %.3f Å (cutoff = %.3f Å)" \\
                  $i0 $j0 $d0 $cutoff]

    set keep [list]
    foreach b $bl {{
        lassign $b i j _t _o
        if {{[measure bond [list $i $j]] <= $cutoff}} {{ lappend keep $b }}
    }}
    set removed [expr {{[llength $bl] - [llength $keep]}}]
    # `both` here is the *flag* (each tuple is {{i j type order}}), NOT a
    # selection; setbondlist replaces VMD's active bond list.
    topo setbondlist both $keep
    puts "viz: removed $removed of [llength $bl] bond(s) longer than $cutoff Å."

    # Fragment + per-fragment-mass summary.  Fragments are the connected
    # components of VMD's bond graph; trimming PBC-crossing bonds usually
    # reveals one large network fragment plus a tail of small bits.
    set all [atomselect top all]
    set frags [$all get fragment]
    set masses [$all get mass]
    $all delete
    set fmass [dict create]
    set fnat  [dict create]
    foreach f $frags m $masses {{
        dict incr fnat $f
        if {{[dict exists $fmass $f]}} {{
            dict set fmass $f [expr {{[dict get $fmass $f] + $m}}]
        }} else {{
            dict set fmass $f $m
        }}
    }}
    set nfrags [dict size $fmass]
    puts "viz: $nfrags fragment(s) after trimming"
    # Sort fragment IDs by mass descending so the big network appears first.
    set pairs [list]
    foreach f [dict keys $fmass] {{
        lappend pairs [list [dict get $fmass $f] $f]
    }}
    set pairs [lsort -decreasing -real -index 0 $pairs]
    set fids [list]
    foreach p $pairs {{ lappend fids [lindex $p 1] }}
    set show_max 20
    set shown 0
    foreach f $fids {{
        if {{$shown >= $show_max && $nfrags > $show_max}} {{
            set rest [expr {{$nfrags - $shown}}]
            puts [format "    ... and %d more fragment(s) (run `htpolynet_viz_fragments` in TkConsole for full list)" $rest]
            break
        }}
        puts [format "    fragment %5d:  %10.2f g/mol   %5d atom(s)" \\
                      $f [dict get $fmass $f] [dict get $fnat $f]]
        incr shown
    }}
    # Stash for later TkConsole querying.
    set ::htpolynet_viz_fmass $fmass
    set ::htpolynet_viz_fnat  $fnat
}}

# Print the full fragment table (call from TkConsole).
proc htpolynet_viz_fragments {{}} {{
    if {{![info exists ::htpolynet_viz_fmass]}} {{
        puts "viz: run htpolynet_trim_pbc_bonds first"
        return
    }}
    foreach f [lsort -integer [dict keys $::htpolynet_viz_fmass]] {{
        puts [format "    fragment %5d:  %10.2f g/mol   %5d atom(s)" \\
                      $f [dict get $::htpolynet_viz_fmass $f] \\
                          [dict get $::htpolynet_viz_fnat  $f]]
    }}
}}

htpolynet_trim_pbc_bonds 3.0
'''


def write_viz_files(top, gro, prefix=None):
    """Generate ``<prefix>.viz.psf`` + ``<prefix>.viz.tcl`` from gromacs top + gro.

    Args:
        top (str): path to a gromacs ``.top`` file.
        gro (str): path to the matching ``.gro`` file.
        prefix (str): output basename.  If None, uses the stem of ``gro``
            (e.g. ``final.gro`` → ``final``).

    Returns:
        tuple[str, str]: paths to the written ``.viz.psf`` and ``.viz.tcl``.

    Raises:
        ImportError: if parmed is not installed.
        Exception: whatever parmed raises if the top file is unparseable.
    """
    import parmed as pmd

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(gro))[0]
    out_dir = os.path.dirname(os.path.abspath(gro)) or '.'
    psf = os.path.join(out_dir, f'{prefix}.viz.psf')
    tcl = os.path.join(out_dir, f'{prefix}.viz.tcl')

    s = pmd.load_file(top, xyz=gro)
    s.save(psf, overwrite=True)
    with open(tcl, 'w') as f:
        f.write(_TCL_TEMPLATE.format(
            psf=os.path.basename(psf),
            gro=os.path.basename(gro),
            tcl=os.path.basename(tcl),
        ))
    logger.info(f'Wrote VMD viz files: {psf}, {tcl}')
    return psf, tcl


def make_viz(args):
    """CLI handler for ``htpolynet make-viz``.

    Args:
        args (argparse.Namespace): parsed CLI arguments with ``.top``, ``.gro``,
            and optional ``.prefix``.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        write_viz_files(args.top, args.gro, prefix=args.prefix)
    except ImportError:
        logger.error('parmed is required for VMD viz generation; '
                     'install with `pip install parmed`.')
        raise SystemExit(1)
    except Exception as e:
        logger.error(f'Failed to generate VMD viz files: {e}')
        raise SystemExit(1)
