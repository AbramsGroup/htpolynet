"""Handles identification of available software needed by HTPolyNet.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import json
import subprocess
import logging
import os
from htpolynet.utils.stringthings import my_logger

logger = logging.getLogger(__name__)

_ambertools = ['antechamber', 'tleap', 'parmchk2']

# AmberTools state
passes = False
versions = {}

# GPU state — populated by _detect_gpus() during sw_setup()
gpu_ids = []

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
    mdrun = _mdrun_cmd(f'{gmx} {gmx_options} mdrun')
    mdrun_single_molecule = _mdrun_cmd(f'{gmx} {gmx_options} mdrun')


def _get_versions():
    """Queries conda for the ambertools package version. Reports
    'installed (version unknown)' if conda is unavailable or the package
    is not listed.
    """
    global versions
    if passes:
        version=None
        try:
            CP=subprocess.run(['conda','list','ambertools','--json'],capture_output=True,text=True)
            pkgs=json.loads(CP.stdout)
            if pkgs:
                version=f'ver. {pkgs[0]["version"]} (conda)'
        except Exception:
            pass
        versions['ambertools']=version or 'installed (version unknown)'
    else:
        versions['ambertools']='Not installed.'


def set_gmx_preferences(parameters):
    """Sets the global Gromacs preferences.

    Args:
        parameters (dict): dictionary from cfg file
    """
    global gmx, gmx_options, mdrun, mdrun_single_molecule
    gromacs_dict=parameters.get('gromacs',{})
    logger.debug(f'gromacs_dict {gromacs_dict}')
    if gromacs_dict:
        gmx=gromacs_dict.get('gmx','gmx')
        gmx_options=gromacs_dict.get('gmx_options','-quiet')
        mdrun=gromacs_dict.get('mdrun',_mdrun_cmd(f'{gmx} {gmx_options} mdrun'))
        mdrun_single_molecule=gromacs_dict.get('mdrun_single_molecule',_mdrun_cmd(f'{gmx} {gmx_options} mdrun'))
        logger.debug(f'{gmx}, {gmx_options}, {mdrun}')
    else:
        gmx_options=parameters.get('gmx_options','')
        gmx=parameters.get('gmx','gmx')
        mdrun=parameters.get('mdrun',_mdrun_cmd(f'{gmx} {gmx_options} mdrun'))
        mdrun_single_molecule=parameters.get('mdrun_single_molecule',_mdrun_cmd(f'{gmx} {gmx_options} mdrun'))
    CP=subprocess.run(['which',gmx],capture_output=True,text=True)
    assert CP.returncode==0,f'{gmx} not found'
    _get_gmx_version()


def _get_gmx_version():
    """Parses the GROMACS version from 'gmx --version' output."""
    global versions
    version = None
    try:
        CP = subprocess.run([gmx, '--version'], capture_output=True, text=True)
        for line in (CP.stdout + CP.stderr).splitlines():
            if 'GROMACS version' in line:
                version = line.split(':',1)[1].strip()
                break
    except Exception:
        pass
    versions['gromacs'] = version or 'installed (version unknown)'


def to_string():
    """Returns a formatted string listing AmberTools, GROMACS, and GPU info."""
    r=['Ambertools:']
    for c in _ambertools:
        r.append(f'{os.path.split(c)[1]:>12s} ({versions.get("ambertools","unknown")})')
    r.append('Gromacs:')
    r.append(f'{"gmx":>12s} ({versions.get("gromacs","unknown")})')
    r.append('GPUs:')
    r.append(f'  {len(gpu_ids)} detected ({", ".join(str(i) for i in gpu_ids) if gpu_ids else "none — mdrun will use CPU"})')
    return '\n'.join(r)


def info():
    """Logs the software version info."""
    my_logger(to_string(), logger.info)
