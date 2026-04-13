"""GROMACS .gro and HTPolyNet .grx file reader/writer.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRO_ATTRIBUTES = [
    'resNum', 'resName', 'atomName', 'globalIdx',
    'posX', 'posY', 'posZ', 'velX', 'velY', 'velZ',
]

GRX_ATTRIBUTES = [
    'z', 'nreactions', 'reactantName', 'sea_idx',
    'bondchain', 'bondchain_idx', 'molecule', 'molecule_name',
]
"""Extended per-atom attributes stored in .grx files.

- z              number of sacrificial H's available on this atom
- nreactions     number of bonds this atom has formed
- reactantName   name of the most recent reactant this atom belonged to
- sea_idx        symmetry-equivalent-atom group index within a residue
- bondchain      index of the C-C bondchain this atom belongs to
- bondchain_idx  position of this atom within its bondchain
- molecule       index of the molecule this atom belongs to
- molecule_name  name of that molecule
"""

GRX_GLOBALLY_UNIQUE = [False, False, False, True, True, False, True, False]
"""Parallel to GRX_ATTRIBUTES: True if the attribute must be globally unique across the system."""

GRX_UNSET_DEFAULTS = [0, 0, 'UNSET', -1, -1, -1, -1, 'UNSET']
"""Parallel to GRX_ATTRIBUTES: sentinel value that means 'not yet assigned'."""


def read(filename):
    """Parse a GROMACS .gro file.

    Args:
        filename (str): path to the .gro file

    Returns:
        dict: parsed data with keys:
            - 'name'    (str): title line
            - 'N'       (int): atom count
            - 'metadat' (dict): metadata (currently just {'N': N})
            - 'A'       (pd.DataFrame): atom data frame
            - 'box'     (np.ndarray, shape 3×3): box matrix
    """
    box = np.zeros((3, 3))
    metadat = {}
    with open(filename, 'r') as f:
        data = f.read().split('\n')
    while '' in data:
        data.remove('')

    name = data[0]
    N = int(data[1])
    metadat['N'] = N

    series = {k: [] for k in GRO_ATTRIBUTES}
    lc_globalIdx = 1
    for x in data[2:-1]:
        series['resNum'].append(int(x[0:5].strip()))
        series['resName'].append(x[5:10].strip())
        series['atomName'].append(x[10:15].strip())
        # globalIdx is always row index + 1 when the file is correctly formatted
        series['globalIdx'].append(lc_globalIdx)
        lc_globalIdx += 1
        # Fixed-width fields; split() fails when columns touch
        # Format: "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"
        numbers = list(map(float, [x[20 + 8 * i:20 + 8 * (i + 1)] for i in range(3)]))
        if len(x) > 44:
            numbers.extend(list(map(float, [x[44 + 8 * i:44 + 8 * (i + 1)] for i in range(3)])))
        series['posX'].append(numbers[0])
        series['posY'].append(numbers[1])
        series['posZ'].append(numbers[2])
        if len(numbers) == 6:
            series['velX'].append(numbers[3])
            series['velY'].append(numbers[4])
            series['velZ'].append(numbers[5])

    if len(series['velX']) == 0:
        del series['velX']
        del series['velY']
        del series['velZ']

    assert N == len(series['globalIdx']), f'Atom count mismatch inside {filename}'
    A = pd.DataFrame(series)

    boxdata = list(map(float, data[-1].split()))
    box[0][0] = boxdata[0]
    box[1][1] = boxdata[1]
    box[2][2] = boxdata[2]
    if len(boxdata) == 9:
        box[0][1], box[0][2], box[1][0], box[1][2], box[2][0], box[2][1] = boxdata[3:]

    return {'name': name, 'N': N, 'metadat': metadat, 'A': A, 'box': box}


def write(filename, A, box, name, grotitle=''):
    """Write a GROMACS .gro file.

    Args:
        filename (str): path to write
        A (pd.DataFrame): atom data frame; must contain the gro_attributes columns
        box (np.ndarray, shape 3×3): box matrix
        name (str): molecule/system name used as the title when grotitle is empty
        grotitle (str): explicit title line, defaults to ''
    """
    title = name if not grotitle else grotitle
    has_vel = 'velX' in A.columns
    gro_cols = GRO_ATTRIBUTES if has_vel else GRO_ATTRIBUTES[:-3]
    atomformatters = (
        [
            lambda x: f'{x:>5d}',
            lambda x: f'{x:<5s}',
            lambda x: f'{x:>5s}',
            lambda x: f'{x % 10000:5d}',
        ]
        + [lambda x: f'{x:8.3f}'] * 3
        + [lambda x: f'{x:8.4f}'] * 3
    )
    with open(filename, 'w') as f:
        f.write(title + '\n')
        f.write(f'{A.shape[0]:>5d}\n')
        for _, r in A.iterrows():
            f.write(''.join([atomformatters[i](v) for i, v in enumerate(list(r[gro_cols]))]) + '\n')
        if not np.any(box):
            logger.debug('Writing Gromacs coordinates file but boxsize is not set.')
        f.write(f'{box[0][0]:10.5f}{box[1][1]:10.5f}{box[2][2]:10.5f}')
        x, y = box.nonzero()
        if not all(x == y):
            f.write(f'{box[0][1]:10.5f}{box[0][2]:10.5f}')
            f.write(f'{box[1][0]:10.5f}{box[1][2]:10.5f}')
            f.write(f'{box[2][0]:10.5f}{box[2][1]:10.5f}')
        f.write('\n')


def write_grx(filename, A, attributes, formatters=[]):
    """Write a subset of atom attributes to an HTPolyNet .grx file.

    The file is a whitespace-delimited table with 'globalIdx' as the first
    column followed by the requested attribute columns.

    Args:
        filename (str): path to write
        A (pd.DataFrame): atom data frame
        attributes (list): attribute names to include
        formatters (list): optional per-column formatter callables, defaults to []

    Raises:
        Exception: if any attribute in attributes is absent from A
    """
    missing = [a for a in attributes if a not in A.columns]
    if missing:
        raise Exception(f'Column(s) {missing} not found in atoms dataframe')
    with open(filename, 'w') as f:
        cols = A[['globalIdx'] + attributes]
        if formatters:
            f.write(cols.to_string(header=True, index=False, formatters=formatters) + '\n')
        else:
            f.write(cols.to_string(header=True, index=False) + '\n')


def read_grx(filename, attributes=[]):
    """Read an HTPolyNet .grx extended-attribute file.

    Args:
        filename (str): path to read
        attributes (list): attribute names to read; if empty, all columns are read

    Returns:
        tuple: (df, attributes_read) where df has a 'globalIdx' column plus the
               attribute columns, and attributes_read is the list of attribute names
               that were actually read.

    Raises:
        AssertionError: if the file does not exist or lacks a 'globalIdx' column
    """
    assert os.path.exists(filename), f'Error: {filename} not found'
    if len(attributes) == 0:
        df = pd.read_csv(filename, sep=r'\s+', header=0)
        assert 'globalIdx' in df.columns, \
            f"Error: {filename} does not have a 'globalIdx' column"
        attributes_read = [c for c in df.columns if c != 'globalIdx']
    else:
        df = pd.read_csv(filename, sep=r'\s+', names=['globalIdx'] + attributes, header=0)
        attributes_read = attributes
    return df, attributes_read
