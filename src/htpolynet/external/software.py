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

# Gromacs state — updated by set_gmx_preferences()
gmx = 'gmx'
gmx_options = '-quiet'
mdrun = f'{gmx} mdrun'
mdrun_single_molecule = f'{gmx} mdrun'

def sw_setup():
    """Checks that all required AmberTools executables are accessible and queries their version."""
    global passes, versions
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
        mdrun=gromacs_dict.get('mdrun',f'{gmx} {gmx_options} mdrun')
        mdrun_single_molecule=gromacs_dict.get('mdrun_single_molecule',f'{gmx} {gmx_options} mdrun')
        logger.debug(f'{gmx}, {gmx_options}, {mdrun}')
    else:
        gmx_options=parameters.get('gmx_options','')
        gmx=parameters.get('gmx','gmx')
        mdrun=parameters.get('mdrun',f'{gmx} {gmx_options} mdrun')
        mdrun_single_molecule=parameters.get('mdrun_single_molecule',f'{gmx} {gmx_options} mdrun')
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
    """Returns a formatted string listing AmberTools and GROMACS versions."""
    r=['Ambertools:']
    for c in _ambertools:
        r.append(f'{os.path.split(c)[1]:>12s} ({versions.get("ambertools","unknown")})')
    r.append('Gromacs:')
    r.append(f'{"gmx":>12s} ({versions.get("gromacs","unknown")})')
    return '\n'.join(r)


def info():
    """Logs the software version info."""
    my_logger(to_string(), logger.info)
