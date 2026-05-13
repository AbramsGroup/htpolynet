"""Handles identification of available software needed by HTPolyNet.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import glob
import json
import logging
import os
import subprocess

from ..utils.stringthings import my_logger

logger = logging.getLogger(__name__)

_ambertools = ['antechamber', 'tleap', 'parmchk2']

# AmberTools state
passes = False
versions = {}

# GPU state — populated by _detect_gpus() during sw_setup()
gpu_ids = []

# HTPolyNet git commit — populated by _get_git_commit() during sw_setup()
git_commit = 'unknown'

# Gromacs state — updated by sw_setup() and set_gmx_preferences()
gmx = 'gmx'
gmx_options = '-quiet'
mdrun = f'{gmx} mdrun'
mdrun_single_molecule = f'{gmx} mdrun'

def _detect_gpus():
    """Populates gpu_ids with the IDs of available NVIDIA GPUs via GPUtil/nvidia-smi.
    Sets gpu_ids to [] if no GPUs are found or nvidia-smi is unavailable.
    """
    global gpu_ids
    try:
        import GPUtil
        gpu_ids = [g.id for g in GPUtil.getGPUs()]
    except Exception:
        gpu_ids = []
    if gpu_ids:
        logger.debug(f'Detected GPUs: {gpu_ids}')
    else:
        logger.debug('No GPUs detected; mdrun will use CPU only')


def _mdrun_cmd(base):
    """Appends appropriate GPU/CPU flags to a base mdrun command string.

    Args:
        base (str): base mdrun command (e.g. 'gmx -quiet mdrun')

    Returns:
        str: command with GPU flags appended if needed
    """
    if gpu_ids:
        return base  # GROMACS auto-selects available GPUs
    else:
        return f'{base} -nb cpu'  # force CPU for non-bonded when no GPU present


def _get_git_commit():
    """Records the short git commit hash of the htpolynet source tree and
    whether there are uncommitted changes.  Falls back to 'unknown' if the
    working directory is not inside a git repo or git is not available.
    """
    global git_commit
    src = os.path.dirname(__file__)
    try:
        cp = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=src,
        )
        if cp.returncode == 0:
            git_commit = cp.stdout.strip()
            dirty = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, cwd=src,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                git_commit += ' (uncommitted changes present)'
    except Exception:
        pass


def sw_setup():
    """Checks that all required AmberTools executables are accessible and queries their version."""
    global passes, versions, mdrun, mdrun_single_molecule
    cnf=[]
    passes=True
    for c in _ambertools:
        CP=subprocess.run(['which',c],capture_output=True,text=True)
        if CP.returncode!=0:
            passes=False
            cnf.append(c)
    if not passes:
        print(f'It seems like you do not have an accessible installation of ambertools.')
    _get_versions()
    _get_gmx_version()
    _detect_gpus()
    _get_git_commit()
    mdrun = _mdrun_cmd(f'{gmx} {gmx_options} mdrun')
    mdrun_single_molecule = _mdrun_cmd(f'{gmx} {gmx_options} mdrun')


def _get_versions():
    """Reads the ambertools package version from conda-meta in the environment
    where antechamber is installed. Avoids running 'conda list', which can fail
    when the host has a broken or mismatched conda on PATH (e.g. inside a container
    on a cluster with a system conda that points to a non-existent environment).
    Falls back to 'installed (version unknown)' if the metadata cannot be found.
    """
    global versions
    if passes:
        version = None
        try:
            cp = subprocess.run(['which', 'antechamber'], capture_output=True, text=True)
            if cp.returncode == 0:
                prefix = os.path.dirname(os.path.dirname(cp.stdout.strip()))
                matches = glob.glob(os.path.join(prefix, 'conda-meta', 'ambertools-*.json'))
                if matches:
                    with open(matches[0]) as f:
                        meta = json.load(f)
                    version = f'ver. {meta["version"]} (conda)'
        except Exception:
            pass
        versions['ambertools'] = version or 'installed (version unknown)'
    else:
        versions['ambertools'] = 'Not installed.'


def set_gmx_preferences(gromacs_dict={}):
    """Sets the global Gromacs preferences.

    Args:
        gromacs_dict (dict): gromacs section from the configuration, defaults to {}
    """
    global gmx, gmx_options, mdrun, mdrun_single_molecule
    logger.debug(f'gromacs_dict {gromacs_dict}')
    if gromacs_dict:
        gmx=gromacs_dict.get('gmx','gmx')
        gmx_options=gromacs_dict.get('gmx_options','-quiet')
        mdrun=gromacs_dict.get('mdrun',_mdrun_cmd(f'{gmx} {gmx_options} mdrun'))
        mdrun_single_molecule=gromacs_dict.get('mdrun_single_molecule',_mdrun_cmd(f'{gmx} {gmx_options} mdrun'))
        logger.debug(f'{gmx}, {gmx_options}, {mdrun}')
    CP=subprocess.run(['which',gmx],capture_output=True,text=True)
    assert CP.returncode==0,f'{gmx} not found'
    _get_gmx_version()
    _enforce_gpu_consistency(gromacs_dict)


def _enforce_gpu_consistency(gromacs_dict):
    """Strip mdrun_options.gpu_id when the runtime cannot honor it.

    A `gpu_id` request is invalid (and will crash gmx mdrun) if either
    (a) the gmx binary was built without GPU support, or (b) no GPU devices
    are visible on the host.  In either case log a warning and remove the
    option so the run can proceed on CPU.
    """
    if not isinstance(gromacs_dict, dict):
        return
    mdrun_options = gromacs_dict.get('mdrun_options')
    if not isinstance(mdrun_options, dict) or 'gpu_id' not in mdrun_options:
        return
    gpu_support = versions.get('gromacs_gpu', 'unknown')
    reasons = []
    if gpu_support.lower() in ('disabled', 'no', 'none', 'off'):
        reasons.append(f'gmx was built with GPU support {gpu_support!r}')
    if not gpu_ids:
        reasons.append('no GPU devices detected on this host')
    if reasons:
        requested = mdrun_options.pop('gpu_id')
        logger.warning(
            f'Removing mdrun_options.gpu_id={requested} from config: '
            + '; '.join(reasons)
            + '. mdrun will run on CPU.'
        )


def _get_gmx_version():
    """Parses the GROMACS version and GPU-support backend from 'gmx --version'."""
    global versions
    version = None
    gpu_support = None
    try:
        CP = subprocess.run([gmx, '--version'], capture_output=True, text=True)
        for line in (CP.stdout + CP.stderr).splitlines():
            if 'GROMACS version' in line and version is None:
                version = line.split(':', 1)[1].strip()
            elif 'GPU support' in line and gpu_support is None:
                gpu_support = line.split(':', 1)[1].strip()
    except Exception:
        pass
    versions['gromacs'] = version or 'installed (version unknown)'
    versions['gromacs_gpu'] = gpu_support or 'unknown'


def to_string():
    """Returns a formatted string listing HTPolyNet commit, AmberTools, GROMACS, and GPU info."""
    r=[f'HTPolyNet git commit: {git_commit}']
    r.append('Ambertools:')
    for c in _ambertools:
        r.append(f'{os.path.split(c)[1]:>12s} ({versions.get("ambertools","unknown")})')
    r.append('Gromacs:')
    r.append(f'{"gmx":>12s} ({versions.get("gromacs","unknown")})')
    r.append(f'{"GPU support":>12s}: {versions.get("gromacs_gpu","unknown")}')
    r.append('GPUs:')
    r.append(f'  {len(gpu_ids)} detected ({", ".join(str(i) for i in gpu_ids) if gpu_ids else "none — mdrun will use CPU"})')
    return '\n'.join(r)


def info():
    """Logs the software version info."""
    my_logger(to_string(), logger.info)
