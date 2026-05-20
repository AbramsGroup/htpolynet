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

# Sidecar selection macros (one per constituent class, one per instance).
# Written when a matching .grx is available next to the .gro.  Defines
# atomselect macros so VMD selections can be driven by chemical identity
# (e.g. `GMA`, `DHT_125`) rather than by the building-block residue names.
if {{[file exists {macros}]}} {{
    source {macros}
}}
'''


def _compress_to_ranges(ints):
    """Compress a list of integers into a VMD-friendly index-selection string.

    Returns ``"0 to 9 15 17 to 19"`` for input ``[0,1,...,9,15,17,18,19]``.
    Sorts and deduplicates the input.

    Args:
        ints (iterable[int]): atom indices (0-based, VMD convention).

    Returns:
        str: space-separated tokens of single indices and ``A to B`` ranges.
    """
    s = sorted(set(int(x) for x in ints))
    if not s:
        return ''
    parts = []
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[j] + 1:
            j += 1
        parts.append(str(s[i]) if i == j else f'{s[i]} to {s[j]}')
        i = j + 1
    return ' '.join(parts)


def write_macros_file(grx, macros_path):
    """Read a .grx and emit a TCL file of VMD atomselect macros.

    For each unique value in the grx ``molecule_name`` column, writes a
    class-level macro (``atomselect macro <NAME> "index ..."``).  For each
    instance (unique pair of ``molecule_name`` and ``molecule``) also writes
    a per-instance macro (``<NAME>_<NNN>``), where ``<NNN>`` is the global
    molecule index zero-padded to a uniform width.

    Atom indices are converted to VMD's 0-based ``index`` convention by
    subtracting 1 from the grx 1-based ``globalIdx``.

    Args:
        grx (str): path to an htpolynet .grx file.
        macros_path (str): path to write the TCL macros file.

    Returns:
        bool: True if the file was written; False if grx lacked the required
            columns or contained no eligible rows.
    """
    import pandas as pd

    df = pd.read_csv(grx, sep=r'\s+', header=0)
    if 'molecule_name' not in df.columns or 'molecule' not in df.columns \
            or 'globalIdx' not in df.columns:
        return False
    df = df[(df['molecule_name'] != 'UNSET') & (df['molecule'] >= 0)].copy()
    if df.empty:
        return False
    df['_vidx'] = df['globalIdx'].astype(int) - 1

    lines = [
        '# htpolynet VMD selection macros — by constituent.',
        '#',
        '# Sourced automatically from the matching .viz.tcl.  Defines two layers',
        '# of atomselect macros so selections can be driven by chemical identity',
        '# rather than by building-block residue names:',
        '#',
        '#   <NAME>            all atoms across every instance of constituent <NAME>',
        '#   <NAME>_<NNN>      all atoms of one specific instance (global molecule index)',
        '#',
        '# Example:',
        '#   mol modselect 0 top GMA       ;# show all bis-GMA molecules',
        '#   mol modselect 0 top DHT_125   ;# show one specific HTPB chain',
        '',
        '# --- constituent classes ---',
    ]
    for cls, sub in df.groupby('molecule_name', sort=True):
        lines.append(f'atomselect macro {cls} "index {_compress_to_ranges(sub["_vidx"].tolist())}"')

    pad = max(3, len(str(int(df['molecule'].max()))))
    lines.append('')
    lines.append('# --- individual instances ---')
    for (cls, mol), sub in df.groupby(['molecule_name', 'molecule'], sort=True):
        macro_name = f'{cls}_{int(mol):0{pad}d}'
        lines.append(f'atomselect macro {macro_name} "index {_compress_to_ranges(sub["_vidx"].tolist())}"')

    with open(macros_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return True


def write_viz_files(top, gro, prefix=None, grx=None):
    """Generate ``<prefix>.viz.psf`` + ``<prefix>.viz.tcl`` from gromacs top + gro.

    If a matching ``.grx`` is found (passed explicitly or auto-discovered next
    to ``gro``), also writes ``<prefix>.viz.macros.tcl`` with VMD atomselect
    macros keyed on constituent identity; the main ``.viz.tcl`` sources it
    when present.

    Args:
        top (str): path to a gromacs ``.top`` file.
        gro (str): path to the matching ``.gro`` file.
        prefix (str): output basename.  If None, uses the stem of ``gro``
            (e.g. ``final.gro`` → ``final``).
        grx (str): path to an htpolynet ``.grx`` file.  If None, looks for
            ``<gro-stem>.grx`` next to ``gro``.

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
    macros = os.path.join(out_dir, f'{prefix}.viz.macros.tcl')

    if grx is None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(gro)) or '.',
                                 os.path.splitext(os.path.basename(gro))[0] + '.grx')
        if os.path.exists(candidate):
            grx = candidate

    macros_written = False
    if grx and os.path.exists(grx):
        try:
            macros_written = write_macros_file(grx, macros)
            if macros_written:
                logger.info(f'Wrote VMD macros: {macros}')
        except Exception as e:
            logger.warning(f'Could not write VMD macros from {grx}: {e}')

    s = pmd.load_file(top, xyz=gro)
    s.save(psf, overwrite=True)
    with open(tcl, 'w') as f:
        f.write(_TCL_TEMPLATE.format(
            psf=os.path.basename(psf),
            gro=os.path.basename(gro),
            tcl=os.path.basename(tcl),
            macros=os.path.basename(macros),
        ))
    logger.info(f'Wrote VMD viz files: {psf}, {tcl}')
    return psf, tcl


def make_viz(args):
    """CLI handler for ``htpolynet make-viz``.

    Args:
        args (argparse.Namespace): parsed CLI arguments with ``.top``, ``.gro``,
            and optional ``.prefix`` and ``.grx``.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        write_viz_files(args.top, args.gro, prefix=args.prefix,
                        grx=getattr(args, 'grx', None))
    except ImportError:
        logger.error('parmed is required for VMD viz generation; '
                     'install with `pip install parmed`.')
        raise SystemExit(1)
    except Exception as e:
        logger.error(f'Failed to generate VMD viz files: {e}')
        raise SystemExit(1)
