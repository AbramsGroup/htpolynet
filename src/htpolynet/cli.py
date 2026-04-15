"""Manages the htpolynet application, provides the command-line interface entry point.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""

import glob
import json
import logging
import os
import tarfile
import textwrap

import argparse as ap

from importlib.metadata import distribution

from .analysis.analyze import analyze
from .analysis.plot import plots
from .analysis.postsim import postsim
from .core import projectfilesystem as pfs
from .core.runtime import Runtime
from .external import software
from .utils.banner import banner_message
from .utils.inputcheck import input_check
from .utils.logsetup import setup_logging
from .utils.stringthings import my_logger

logger = logging.getLogger(__name__)


def _is_editable_install():
    """Returns True if htpolynet was installed in editable mode (pip install -e .)."""
    try:
        direct_url = distribution('htpolynet').read_text('direct_url.json')
        if direct_url:
            return json.loads(direct_url).get('dir_info', {}).get('editable', False)
    except Exception:
        pass
    return False


def info(args: ap.Namespace):
    """Handles the info subcommand.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    print('This is some information on your installed version of htpolynet')
    l = pfs.lib_setup()
    software.sw_setup()
    print(l.info())
    print(software.to_string())
    system_mols, cached_mols = pfs.get_molecule_info()
    print('Available molecules in system library:')
    for m in system_mols:
        print(f'   {m}')
    if cached_mols:
        print('Cached parameterized molecules:')
        for m in cached_mols:
            print(f'   {m}')
    possibles = l.get_example_names()
    print('Available examples using htpolynet fetch-example')
    for p in possibles:
        if p[0].isdecimal():
            print(f'   {p}')

def run(args: ap.Namespace):
    """Handles the run subcommand.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    setup_logging(args.loglevel, diag=args.diag, no_banner=args.no_banner)
    my_logger('htpolynet runtime begins', logger.info)
    userlib = args.lib if os.path.exists(args.lib) else None
    software.sw_setup()
    software.info()
    pfs.pfs_setup(root=os.getcwd(), topdirs=pfs.Dirs.run_topdirs, verbose=True, projdir=args.proj, reProject=args.restart, userlibrary=userlib)
    a = Runtime(cfgfile=args.config, restart=args.restart)
    if args.param_only:
        a.generate_molecules(force_checkin=args.force_checkin, force_parameterization=args.force_parameterization)
    else:
        a.do_workflow(force_checkin=args.force_checkin, force_parameterization=args.force_parameterization)
    my_logger('htpolynet runtime ends', logger.info)

def _unpack_example(fullpath):
    """Extracts a single example tarball into the current directory.

    Args:
        fullpath (str): absolute path to the .tgz file in the system library
    """
    with tarfile.open(fullpath) as tf:
        tf.extractall('.')

def fetch_example(args):
    """Handles the fetch-example subcommand.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    l = pfs.system()
    depot = l.get_example_depot_location()
    possibles = l.get_example_names()
    if args.n == 'all':
        for fullname in possibles:
            _unpack_example(os.path.join(depot, f'{fullname}.tgz'))
        return
    if args.n.isdecimal():
        fullname = [x for x in possibles if x.startswith(args.n)][0]
    else:
        fullname = args.n
    fullpath = os.path.join(depot, f'{fullname}.tgz')
    assert os.path.exists(fullpath), f'{fullpath}: not found.'
    _unpack_example(fullpath)

def pack_example(args):
    l = pfs.system()
    requires = ['README.md', 'run.sh', 'lib/molecules']
    for r in requires:
        assert os.path.exists(r), f'{r}: not found.'
    ls = glob.glob('*.yaml') + glob.glob('*.yml')
    assert len(ls)>0, f'No yaml config file found.'
    bn = os.path.basename(os.getcwd())
    inspectname = bn.split('-')
    if len(inspectname) > 0:
        firstfield = inspectname[0]
        if firstfield.isdigit():
            existing_n = int(firstfield)
            use_n = False
        else:
            use_n = True
    n = args.n
    overwrite = args.overwrite
    existing_examples = l.get_example_names()
    numbers = [int(x.split('-')[0]) for x in existing_examples]
    if not overwrite:
        assert not n in numbers, f'Choose an index other than {n}; an example with this index already exists.'
        if not use_n:
            assert not existing_n in numbers, f'Choose an index other than {n}; an example with this index already exists. To do this, rename the directory.'
    if n == -1:
        n = str(max(numbers) + 1)
    if use_n:
        newname = f'{n}-{os.path.basename(os.getcwd())}'
    else:
        newname = f'{os.path.basename(os.getcwd())}'
    depot_location = l.get_example_depot_location()
    outpath = os.path.join(depot_location, f'{newname}.tgz')
    if overwrite and os.path.exists(outpath):
        logger.debug(f'Warning: overwriting example {outpath}')

    def _shallow(tarinfo):
        # mirror --exclude="*/*/*/*/*": omit paths deeper than 4 components
        from pathlib import Path
        return tarinfo if len(Path(tarinfo.name).parts) <= 4 else None

    with tarfile.open(outpath, 'w:gz') as tf:
        for fname in ['README.md', 'run.sh'] + ls:
            tf.add(fname, arcname=f'{bn}/{fname}', recursive=False)
        tf.add('lib/molecules', arcname=f'{bn}/lib/molecules', filter=_shallow)
    logger.debug(f'Packed {outpath} -- consider a pull request!')

def _add_runtime_args(p, loglevel='debug'):
    """Adds the common arguments shared by the run and parameterize subcommands."""
    p.add_argument('config',type=str,default=None,help='input configuration file in YAML format')
    p.add_argument('-lib',type=str,default='lib',help='local user library of molecular structures and parameterizations')
    p.add_argument('-proj',type=str,default='next',help='project directory; "next" generates the next available; otherwise creates or resumes (with -restart) a named directory')
    p.add_argument('-diag',type=str,default='htpolynet_runtime_diagnostics.log',help='diagnostic log file')
    p.add_argument('-restart',default=False,action='store_true',help='restart in latest proj dir')
    p.add_argument('--no-banner',default=False,action='store_true',help='suppress the startup banner')
    p.add_argument('--force-parameterization',default=False,action='store_true',help='force GAFF parameterization of any input mol2 structures')
    p.add_argument('--force-checkin',default=False,action='store_true',help='force check-in of generated parameter files to the system library')
    p.add_argument('--param-only',default=False,action='store_true',help='stop after parameterizing molecules without building the full system')
    p.add_argument('--loglevel',type=str,default=loglevel,help='log level for the diagnostic file (debug|info)')

def _add_analysis_args(p):
    """Adds the common arguments shared by the postsim and analyze subcommands."""
    p.add_argument('-proj',type=str,default='',nargs='+',help='name of project directory')
    p.add_argument('-lib',type=str,default='lib',help='local user library of molecular structures and parameterizations')
    p.add_argument('-ocfg',type=str,default='',help='original htpolynet config file used to generate project(s)')
    p.add_argument('-cfg',type=str,default='',help='config file specifying the operations to perform')
    p.add_argument('--no-banner',default=False,action='store_true',help='suppress the startup banner')
    p.add_argument('--loglevel',type=str,default='info',help='log level for the diagnostic file (debug|info)')

def cli():
    """Command-line interface."""
    l = pfs.lib_setup()
    example_names = l.get_example_names()
    example_ids = [x.split('-')[0] for x in example_names]

    subcommands = [
        ('run',         run,          'build a system using instructions in the config file and any required molecular structure inputs (use --param-only to stop after parameterization)'),
        ('info',        info,         'print version, software, and library information'),
        ('plots',       plots,        'generate plots summarizing aspects of the current completed build'),
        ('fetch-example',fetch_example,'fetch and unpack example(s): '+', '.join(f'"{x}"' for x in example_names)),
        ('input-check', input_check,  'report the number of atoms that would be in the initial system based on config'),
        ('postsim',     postsim,      'perform specified post-cure MD simulations on final results in one or more project directories'),
        ('analyze',     analyze,      "perform 'gmx <command>' style analyses specified in the config file"),
    ]
    if _is_editable_install():
        subcommands.append(('pack-example', pack_example, '(dev) pack current directory as a tarball into resources/example_depot'))

    parser = ap.ArgumentParser(description=textwrap.dedent(banner_message),formatter_class=ap.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers()
    subparsers.required = True
    cp = {}
    for name,handler,help_str in subcommands:
        cp[name] = subparsers.add_parser(name,help=help_str)
        cp[name].set_defaults(func=handler)

    _add_runtime_args(cp['run'])

    cp['plots'].add_argument('source', type=str, choices=['diag', 'build', 'post'], default='build', help='source of data to plot')
    cp['plots'].add_argument('--diags', type=str, default=[], nargs='+', help='names of diagnostic log files (1 or more)')
    cp['plots'].add_argument('--proj', nargs='+', type=str, default=[], help='name of project director[y/ies]')
    cp['plots'].add_argument('--cfg', type=str, nargs='+', default=[], help='name of input config files')
    cp['plots'].add_argument('--buildplot', type=str, nargs='+', default=['t'], choices=['t', 'g', 'n', 'c'], help='build plot type: t=traces, g=2-D graph, n=homo-N between crosslinks, c=cluster-size distributions')
    cp['plots'].add_argument('--traces', type=str, nargs='+', default=['t', 'd', 'p'], choices=['t', 'd', 'p'], help='trace types: t=temperature, d=density, p=potential energy')
    cp['plots'].add_argument('--n_points', type=int, nargs=2, default=[10, 20], help='number of [cold-side,hot-side] data points for Tg line fits')
    cp['plots'].add_argument('--plotfile', type=str, default='', help='name of output plot file')
    cp['plots'].add_argument('--no-banner', default=False, action='store_true', help='suppress the startup banner')
    cp['plots'].add_argument('--loglevel', type=str, default='info', help='log level for the diagnostic file (debug|info)')

    cp['fetch-example'].add_argument('n', type=str, choices=example_ids+['all'], help='example to fetch: '+', '.join(example_names))

    cp['input-check'].add_argument('config', type=str, default=None, help='input configuration file in YAML format')
    cp['input-check'].add_argument('-lib', type=str, default='lib', help='local user library of molecular structures and parameterizations')

    _add_analysis_args(cp['postsim'])
    _add_analysis_args(cp['analyze'])

    if 'pack-example' in cp:
        cp['pack-example'].add_argument('-n', type=int, default=-1, help='desired index for this example (default: next available)')
        cp['pack-example'].add_argument('--overwrite', default=False, action='store_true', help='overwrite any existing example with this index in the depot')

    args = parser.parse_args()
    args.func(args)
