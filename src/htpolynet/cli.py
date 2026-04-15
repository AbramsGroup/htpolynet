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
import yaml

from importlib.metadata import distribution

from .analysis.analyze import analyze
from .analysis.plot import plots
from .analysis.postsim import postsim
from .core import projectfilesystem as pfs
from .core.runtime import Runtime
from .external import software
from .external.slurm import generate_script, _normalize_keys
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
    my_logger(software.to_string(), logger.info)
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

def _fetch_one(depot, fullname):
    """Fetches one example by name, preferring a .sh script over a .tgz tarball.

    Args:
        depot (str): path to the example depot directory
        fullname (str): example name without extension
    """
    import shutil
    sh_path = os.path.join(depot, f'{fullname}.sh')
    tgz_path = os.path.join(depot, f'{fullname}.tgz')
    if os.path.exists(sh_path):
        dest = os.path.basename(sh_path)
        shutil.copy(sh_path, dest)
        print(f'Fetched {dest}  (run with: bash {dest})')
    elif os.path.exists(tgz_path):
        _unpack_example(tgz_path)
    else:
        raise FileNotFoundError(f'No example found for {fullname}')

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
            _fetch_one(depot, fullname)
        return
    if args.n.isdecimal():
        fullname = [x for x in possibles if x.startswith(args.n)][0]
    else:
        fullname = args.n
    _fetch_one(depot, fullname)

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

def pack_example_sh(args):
    """Generates a self-contained bash setup script for the current example directory.

    Scans the current directory for YAML configs and lib/molecules/inputs/ files,
    embeds them as heredocs, and writes the resulting .sh script to the example depot.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    import textwrap as _tw

    l = pfs.system()
    yamls = sorted(glob.glob('*.yaml') + glob.glob('*.yml'))
    assert yamls, 'No YAML config file found in the current directory.'
    inputs_dir = 'lib/molecules/inputs'
    assert os.path.isdir(inputs_dir), f'{inputs_dir}: not found.'

    # Collect text-format monomer input files (mol2, pdb); skip binaries like png
    input_files = sorted(
        f for f in glob.glob(os.path.join(inputs_dir, '*'))
        if os.path.splitext(f)[1].lower() in ('.mol2', '.pdb')
    )

    bn = os.path.basename(os.getcwd())
    inspectname = bn.split('-')
    firstfield = inspectname[0] if inspectname else ''
    use_n = not firstfield.isdigit()
    existing_n = int(firstfield) if not use_n else None

    n = args.n
    overwrite = args.overwrite
    existing_examples = l.get_example_names()
    numbers = [int(x.split('-')[0]) for x in existing_examples]
    if not overwrite:
        if n != -1:
            assert n not in numbers, f'Index {n} already exists; use --overwrite or choose another.'
        if not use_n:
            assert existing_n not in numbers, (
                f'Index {existing_n} (from directory name) already exists; '
                f'use --overwrite or rename the directory.'
            )
    if n == -1:
        n = max(numbers) + 1 if numbers else 0
    newname = f'{n}-{bn}' if use_n else bn
    display_n = n if use_n else existing_n

    depot_location = l.get_example_depot_location()
    outpath = os.path.join(depot_location, f'{newname}.sh')
    if overwrite and os.path.exists(outpath):
        logger.debug(f'Warning: overwriting {outpath}')

    # Determine the primary config (first yaml alphabetically) for the --run line
    primary_yaml = yamls[0]

    lines = [
        '#!/bin/bash',
        f'# htpolynet -- Example {display_n} -- {bn}',
        '#',
        f'# Writes all input files into ./{newname}/ and optionally launches the build.',
        '#',
        '# Usage:',
        f'#   bash {newname}.sh          # set up directory only',
        f'#   bash {newname}.sh --run    # set up and launch htpolynet',
        '#',
        '# Generated by: htpolynet pack-example-sh',
        '#',
        '# Cameron F. Abrams — cfa22@drexel.edu',
        '',
        'set -euo pipefail',
        '',
        f'EXDIR="{newname}"',
        'if [ -d "$EXDIR" ]; then',
        '    echo "Error: directory \'$EXDIR\' already exists. Remove it first." >&2',
        '    exit 1',
        'fi',
        '',
        'mkdir -p "$EXDIR/lib/molecules/inputs" "$EXDIR/lib/molecules/parameterized"',
        'cd "$EXDIR"',
        '',
    ]

    # Embed each monomer input file as a heredoc
    if input_files:
        lines.append('# ---------------------------------------------------------------------------')
        lines.append('# Monomer input files')
        lines.append('# ---------------------------------------------------------------------------')
        for fpath in input_files:
            dest = os.path.join('lib/molecules/inputs', os.path.basename(fpath))
            token = f'HTPOLYNET_{os.path.basename(fpath).replace(".", "_").upper()}_EOF'
            lines.append(f"cat > {dest} << '{token}'")
            with open(fpath) as fh:
                lines.append(fh.read().rstrip())
            lines.append(token)
            lines.append('')

    # Embed each YAML config as a heredoc
    lines.append('# ---------------------------------------------------------------------------')
    lines.append('# Configuration file(s)')
    lines.append('# ---------------------------------------------------------------------------')
    for yaml_file in yamls:
        token = f'HTPOLYNET_{yaml_file.replace(".", "_").upper()}_EOF'
        lines.append(f"cat > {yaml_file} << '{token}'")
        with open(yaml_file) as fh:
            lines.append(fh.read().rstrip())
        lines.append(token)
        lines.append('')

    # --run launcher
    lines += [
        '# ---------------------------------------------------------------------------',
        '# Optionally launch the build',
        '# ---------------------------------------------------------------------------',
        'if [[ "${1:-}" == "--run" ]]; then',
        f'    htpolynet run -diag diagnostics.log {primary_yaml}',
        'else',
        '    echo "Setup complete. To run:"',
        f'    echo "  cd $EXDIR && htpolynet run -diag diagnostics.log {primary_yaml}"',
        'fi',
    ]

    script = '\n'.join(lines) + '\n'
    with open(outpath, 'w') as fh:
        fh.write(script)
    logger.debug(f'Wrote {outpath} -- consider a pull request!')
    print(f'Wrote {outpath}')


def _add_run_options(p, loglevel='debug'):
    """Adds run options (no positional config) shared by run and gen-slurm-script."""
    p.add_argument('-lib',type=str,default='lib',help='local user library of molecular structures and parameterizations')
    p.add_argument('-proj',type=str,default='next',help='project directory; "next" generates the next available; otherwise creates or resumes (with -restart) a named directory')
    p.add_argument('-diag',type=str,default='htpolynet_runtime_diagnostics.log',help='diagnostic log file')
    p.add_argument('-restart',default=False,action='store_true',help='restart in latest proj dir')
    p.add_argument('--no-banner',default=False,action='store_true',help='suppress the startup banner')
    p.add_argument('--force-parameterization',default=False,action='store_true',help='force GAFF parameterization of any input mol2 structures')
    p.add_argument('--force-checkin',default=False,action='store_true',help='force check-in of generated parameter files to the system library')
    p.add_argument('--param-only',default=False,action='store_true',help='stop after parameterizing molecules without building the full system')
    p.add_argument('--loglevel',type=str,default=loglevel,help='log level for the diagnostic file (debug|info)')


def _add_runtime_args(p, loglevel='debug'):
    """Adds the common arguments shared by the run and parameterize subcommands."""
    p.add_argument('config',type=str,default=None,help='input configuration file in YAML format')
    _add_run_options(p, loglevel=loglevel)

def gen_slurm_script(args: ap.Namespace):
    """Handles the gen-slurm-script subcommand.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    # Load YAML config to pick up the optional slurm: section
    slurm_cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        slurm_cfg = _normalize_keys(cfg.get('slurm', {}))

    # CLI overrides take precedence over values from the YAML slurm: section
    cli_overrides = {
        'job_name':        args.job_name,
        'partition':       args.partition,
        'account':         args.account,
        'nodes':           args.nodes,
        'ntasks':          args.ntasks,
        'ntasks_per_node': args.ntasks_per_node,
        'cpus_per_task':   args.cpus_per_task,
        'time':            args.time,
        'mem':             args.mem,
        'gres':            args.gres,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            slurm_cfg[k] = v

    # --sif / --apptainer-alias override YAML values
    sif = args.sif or slurm_cfg.get('sif')
    apptainer_alias = args.apptainer_alias or slurm_cfg.get('apptainer_alias', 'apptainer')

    if not args.native and not sif:
        raise SystemExit(
            'error: containerized mode requires --sif <path/to/image.sif> '
            '(or a sif: entry in the slurm: section of the config file)'
        )

    run_args = {
        'lib':                    args.lib,
        'proj':                   args.proj,
        'diag':                   args.diag,
        'restart':                args.restart,
        'no_banner':              args.no_banner,
        'force_parameterization': args.force_parameterization,
        'force_checkin':          args.force_checkin,
        'param_only':             args.param_only,
        'loglevel':               args.loglevel,
    }

    script = generate_script(
        config_file=args.config,
        slurm_cfg=slurm_cfg,
        run_args=run_args,
        native=args.native,
        sif=sif,
        apptainer_alias=apptainer_alias,
    )

    outfile = args.output or f'submit_{os.path.splitext(os.path.basename(args.config))[0]}.sh'
    with open(outfile, 'w') as fh:
        fh.write(script)
    print(f'Wrote SLURM script: {outfile}')


def _add_gen_slurm_args(p):
    """Adds arguments for the gen-slurm-script subcommand."""
    p.add_argument('config', type=str, help='htpolynet YAML config file')
    # run options (mirrors _add_run_options)
    _add_run_options(p, loglevel='info')
    # execution mode
    p.add_argument('--native', default=False, action='store_true',
                   help='run htpolynet natively instead of via Apptainer container')
    p.add_argument('--sif', type=str, default=None,
                   help='path to the Apptainer/Singularity .sif image (required unless --native)')
    p.add_argument('--apptainer-alias', type=str, default=None, dest='apptainer_alias',
                   help='container runtime command (default: apptainer)')
    # output
    p.add_argument('-o', '--output', type=str, default=None,
                   help='output script filename (default: submit_<config>.sh)')
    # common SLURM directives as CLI overrides
    p.add_argument('--job-name',        type=str, default=None, dest='job_name')
    p.add_argument('--partition',       type=str, default=None)
    p.add_argument('--account',         type=str, default=None)
    p.add_argument('--nodes',           type=int, default=None)
    p.add_argument('--ntasks',          type=int, default=None)
    p.add_argument('--ntasks-per-node', type=int, default=None, dest='ntasks_per_node')
    p.add_argument('--cpus-per-task',   type=int, default=None, dest='cpus_per_task')
    p.add_argument('--time',            type=str, default=None)
    p.add_argument('--mem',             type=str, default=None)
    p.add_argument('--gres',            type=str, default=None)


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
        ('run',              run,              'build a system using instructions in the config file and any required molecular structure inputs (use --param-only to stop after parameterization)'),
        ('info',             info,             'print version, software, and library information'),
        ('plots',            plots,            'generate plots summarizing aspects of the current completed build'),
        ('fetch-example',    fetch_example,    'fetch and unpack example(s): '+', '.join(f'"{x}"' for x in example_names)),
        ('input-check',      input_check,      'report the number of atoms that would be in the initial system based on config'),
        ('postsim',          postsim,          'perform specified post-cure MD simulations on final results in one or more project directories'),
        ('analyze',          analyze,          "perform 'gmx <command>' style analyses specified in the config file"),
        ('gen-slurm-script', gen_slurm_script, 'generate a SLURM submission script for running htpolynet on a cluster'),
    ]
    if _is_editable_install():
        subcommands.append(('pack-example', pack_example, '(dev) pack current directory as a tarball into resources/example_depot'))
        subcommands.append(('pack-example-sh', pack_example_sh, '(dev) generate a self-contained bash setup script from the current example directory'))

    parser = ap.ArgumentParser(description=textwrap.dedent(banner_message),formatter_class=ap.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers()
    subparsers.required = True
    cp = {}
    for name,handler,help_str in subcommands:
        cp[name] = subparsers.add_parser(name,help=help_str)
        cp[name].set_defaults(func=handler)

    _add_runtime_args(cp['run'])
    _add_gen_slurm_args(cp['gen-slurm-script'])

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
    if 'pack-example-sh' in cp:
        cp['pack-example-sh'].add_argument('-n', type=int, default=-1, help='desired index for this example (default: next available)')
        cp['pack-example-sh'].add_argument('--overwrite', default=False, action='store_true', help='overwrite any existing example with this index in the depot')

    args = parser.parse_args()
    args.func(args)
