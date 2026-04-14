"""Manages execution of AmberTools antechamber, parmchk2.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import hashlib
import logging
import os
import shutil

import parmed

from ..core.coordinates import Coordinates
from ..external.command import run

logger = logging.getLogger(__name__)


def GAFFParameterize(inputPrefix, outputPrefix, input_structure_format='mol2', ambertools=None):
    """Manages execution of antechamber, parmchk2, and tleap to generate GAFF parameters,
    then converts the result to Gromacs gro/top files via parmed.

    Args:
        inputPrefix (str): basename of input structure file
        outputPrefix (str): basename of output files
        input_structure_format (str): format of input structure file, defaults to 'mol2'; 'pdb' is other option
        ambertools (dict | None): ambertools configuration directives, defaults to None

    Raises:
        parmed.exceptions.GromacsError: if parmed fails to convert tleap output
    """
    ambertools = ambertools or {}
    chargemethod = ambertools.get('charge_method', 'bcc')
    logger.info(f'AmberTools> generating GAFF parameters from {inputPrefix}.{input_structure_format}')

    structin  = f'{inputPrefix}.{input_structure_format}'
    mol2out   = f'{outputPrefix}.mol2'
    frcmodout = f'{outputPrefix}.frcmod'
    groOut    = f'{outputPrefix}.gro'
    topOut    = f'{outputPrefix}.top'
    itpOut    = f'{outputPrefix}.itp'

    # antechamber overwrites its input when input and output share the same path
    new_structin = structin
    if structin == mol2out:
        new_structin = f'{inputPrefix}-input.{input_structure_format}'
        logger.debug(f'AmberTools> Antechamber overwrites input {structin}; backing up to {new_structin}')
        shutil.copy(structin, new_structin)

    run(f'antechamber -j 4 -fi {input_structure_format} -fo mol2 -c {chargemethod}'
        f' -at gaff -i {new_structin} -o {mol2out} -pf Y -nc 0 -eq 1 -pl 10', quiet=False)
    logger.debug(f'AmberTools> Antechamber generated {mol2out}')
    run(f'parmchk2 -i {mol2out} -o {frcmodout} -f mol2 -s gaff', quiet=False)

    # Antechamber ignores SUBSTRUCTURE records, so patch the antechamber output
    # mol2 with the original resName/resNum before passing it to tleap.
    if input_structure_format == 'mol2':
        orig    = Coordinates.read_mol2(new_structin)
        ac_out  = Coordinates.read_mol2(mol2out)
        ac_out.A[['resName', 'resNum']] = orig.A[['resName', 'resNum']]
        leap_coords = ac_out
    else:
        leap_coords = Coordinates.read_mol2(mol2out)

    # tleap rejects filenames containing 'e' (treats them as scientific notation),
    # so hash the prefix to get a safe temporary name.
    leapprefix = hashlib.shake_128(outputPrefix.encode('utf-8')).hexdigest(16).replace('e', 'x')
    logger.debug(f'Replacing "{outputPrefix}" with hash "{leapprefix}" for tleap input files.')
    leap_coords.write_mol2(f'{leapprefix}.mol2')
    shutil.copy(frcmodout, f'{leapprefix}.frcmod')

    with open(f'{inputPrefix}-tleap.in', 'w') as f:
        f.write('\n'.join([
            'source leaprc.gaff',
            f'mymol = loadmol2 {leapprefix}.mol2',
            'check mymol',
            f'loadamberparams {leapprefix}.frcmod',
            f'saveamberparm mymol {leapprefix}-tleap.top {leapprefix}-tleap.crd',
            'quit',
            '',
        ]))

    run(f'tleap -f {inputPrefix}-tleap.in', override=('Error!', 'Unspecified tleap error'))

    os.remove(f'{leapprefix}.frcmod')
    shutil.move(f'{leapprefix}.mol2',      mol2out)
    shutil.move(f'{leapprefix}-tleap.top', f'{outputPrefix}-tleap.top')
    shutil.move(f'{leapprefix}-tleap.crd', f'{outputPrefix}-tleap.crd')

    try:
        file = parmed.load_file(f'{outputPrefix}-tleap.top', xyz=f'{outputPrefix}-tleap.crd')
        logger.debug(f'Writing {groOut}, {topOut}, and {itpOut}')
        file.save(groOut, overwrite=True)
        file.save(topOut, parameters=itpOut, overwrite=True)
    except Exception as m:
        logger.error('Unspecified parmed error')
        raise parmed.exceptions.GromacsError(m) from m

