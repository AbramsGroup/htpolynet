"""PDB file reader for ATOM/HETATM coordinates and CONECT-record bond orders.

Interprets repeated CONECT entries for a given atom pair as higher-order bonds:
  a partner listed twice  → double bond (order '2')
  a partner listed three times → triple bond (order '3')

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from ..geometry.bondlist import Bondlist
from ..io.mol2 import MOL2_ATOM_ATTRIBUTES, MOL2_BOND_ATTRIBUTES, MOL2_BOND_TYPES

logger = logging.getLogger(__name__)

# Fallback SYBYL types by element; antechamber will reassign GAFF types.
_ELEMENT_TO_SYBYL = {
    'C':  'C.3',
    'N':  'N.3',
    'O':  'O.3',
    'S':  'S.3',
    'P':  'P.3',
    'H':  'H',
    'F':  'F',
    'CL': 'Cl',
    'BR': 'Br',
    'I':  'I',
}


def _element_from_pdb_line(line, atom_name):
    """Return the element symbol from columns 77-78 or by stripping digits from atom_name."""
    if len(line) >= 78:
        elem = line[76:78].strip()
        if elem:
            return elem.capitalize()
    # fallback: first alphabetic character(s) in atom_name
    stripped = ''.join(c for c in atom_name if c.isalpha())
    return stripped[:2].capitalize() if len(stripped) >= 2 else stripped[:1].upper()


def read(filename):
    """Parse a PDB file into coordinate and bond data compatible with mol2 conventions.

    Bond orders are inferred from CONECT records: if the same bonded-atom serial
    appears N times in one CONECT record, that bond has order N.

    Positions are converted from Ångström (PDB) to nm (htpolynet internal).

    Args:
        filename (str): path to the PDB file

    Returns:
        dict with keys:
            'name'          (str)
            'N'             (int): atom count
            'metadat'       (dict): mol2-compatible metadata
            'A'             (pd.DataFrame): atom data with MOL2_ATOM_ATTRIBUTES columns
            'mol2_bonds'    (pd.DataFrame): bond table with MOL2_BOND_ATTRIBUTES columns
            'mol2_bondlist' (Bondlist)
    """
    atoms = []
    # bond_order[(ai, aj)] = max repetitions seen from either atom's CONECT record, ai < aj
    bond_order = defaultdict(int)

    with open(filename) as f:
        for line in f:
            record = line[:6].strip()

            if record in ('ATOM', 'HETATM'):
                serial  = int(line[6:11])
                name    = line[12:16].strip()
                resname = line[17:20].strip()
                resseq  = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = _element_from_pdb_line(line, name)
                sybyl = _ELEMENT_TO_SYBYL.get(elem.upper(), elem)
                atoms.append({
                    'globalIdx': serial,
                    'atomName':  name,
                    'posX':      x * 0.1,
                    'posY':      y * 0.1,
                    'posZ':      z * 0.1,
                    'type':      sybyl,
                    'resNum':    resseq,
                    'resName':   resname,
                    'charge':    0.0,
                })

            elif record == 'CONECT':
                src = int(line[6:11])
                partners = []
                for col in [11, 16, 21, 26]:
                    if len(line) > col + 4:
                        fld = line[col:col + 5].strip()
                        if fld:
                            try:
                                partners.append(int(fld))
                            except ValueError:
                                pass
                # Count how many times each partner appears in this line
                per_partner = defaultdict(int)
                for p in partners:
                    per_partner[p] += 1
                for p, cnt in per_partner.items():
                    key = (min(src, p), max(src, p))
                    bond_order[key] = max(bond_order[key], cnt)

    if not atoms:
        raise ValueError(f'No ATOM/HETATM records found in {filename}')

    A = pd.DataFrame(atoms)

    bonds = []
    for idx, ((ai, aj), order) in enumerate(sorted(bond_order.items()), 1):
        bonds.append({
            'bondIdx': idx,
            'ai':      ai,
            'aj':      aj,
            'order':   str(order),
        })

    mol2_bonds = pd.DataFrame(bonds).astype(MOL2_BOND_TYPES) if bonds else pd.DataFrame(
        columns=MOL2_BOND_ATTRIBUTES
    )

    if bonds:
        logger.debug(f'pdb.read {filename}: {len(atoms)} atoms, {len(bonds)} bonds '
                     f'({sum(1 for b in bonds if b["order"] != "1")} non-single)')

    mol2_bondlist = Bondlist.fromDataFrame(mol2_bonds)

    name = atoms[0]['resName'] if atoms else ''
    metadat = {
        'N':              len(atoms),
        'nBonds':         len(bonds),
        'nSubs':          1,
        'nFeatures':      0,
        'nSets':          0,
        'mol2type':       'SMALL',
        'mol2chargetype': 'GASTEIGER',
    }

    return {
        'name':          name,
        'N':             len(atoms),
        'metadat':       metadat,
        'A':             A,
        'mol2_bonds':    mol2_bonds,
        'mol2_bondlist': mol2_bondlist,
    }
