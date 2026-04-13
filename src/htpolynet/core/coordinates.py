"""Class for managing atom coordinate data.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""

import logging
import os

import numpy as np
import pandas as pd

from ..io import gro as _gro_io
from ..io import mol2 as _mol2_io

from ..geometry.bondlist import Bondlist
from ..geometry.linkcell import Linkcell
from ..geometry.matrix4 import Matrix4
from ..geometry.ring import Ring, Segment
from ..io.gro import GRX_ATTRIBUTES, GRX_GLOBALLY_UNIQUE, GRX_UNSET_DEFAULTS
from ..utils.dataframetools import get_rows_w_attribute, set_row_attribute, set_rows_attributes_from_dict

logger = logging.getLogger(__name__)


class Coordinates:
    """ Handles atom coordinates.

    The primary object is `A`, a pandas DataFrame with one row per atom.  Each atom has attributes that may be found in a `gro` file and/or a `mol2` file, along with so-called extended attributes, which are used solely by HTPolyNet.  

    """
    gro_attributes = _gro_io.GRO_ATTRIBUTES
    mol2_atom_attributes = _mol2_io.MOL2_ATOM_ATTRIBUTES
    mol2_bond_attributes = _mol2_io.MOL2_BOND_ATTRIBUTES
    mol2_bond_types = _mol2_io.MOL2_BOND_TYPES

    def __init__(self, name=''):
        """Constructs an empty Coordinates object.

        Args:
            name (str): a name string, defaults to ''
        """
        self.name = name
        self.metadat = {}
        self.N = 0
        self.A = pd.DataFrame()
        self.mol2_bonds = pd.DataFrame()
        self.mol2_bondlist = Bondlist()
        self.linkcell = Linkcell(pbc_wrapper=self.wrap_point)
        self.empty = True
        self.box = np.zeros((3,3))
        self.grx_attributes = GRX_ATTRIBUTES
        self.parent = None
        
    @classmethod
    def read_gro(cls, filename, wrap_coords=True):
        """Reads a Gromacs gro file.

        Args:
            filename (str): name of gro file
            wrap_coords (bool): if True, wrap atom coordinates into the box, defaults to True

        Returns:
            Coordinates: a new Coordinates instance
        """
        inst = cls(filename)
        if filename != '':
            data = _gro_io.read(filename)
            inst.name = data['name']
            inst.N = data['N']
            inst.metadat = data['metadat']
            inst.A = data['A']
            inst.box = data['box']
            inst.empty = False
            if wrap_coords:
                inst.wrap_coords()
        return inst

    @classmethod
    def read_mol2(cls, filename):
        """Reads in a Sybyl MOL2 file into a Coordinates instance.

        Note that this method only reads in MOLECULE, ATOM, and BOND sections.
        All lengths are converted from Ångström to nm.

        Args:
            filename (str): name of input mol2 file

        Returns:
            Coordinates: a new Coordinates instance
        """
        inst = cls(name=filename)
        data = _mol2_io.read(filename)
        inst.name = data['name']
        inst.N = data['N']
        inst.metadat = data['metadat']
        inst.A = data['A']
        inst.mol2_bonds = data['mol2_bonds']
        inst.mol2_bondlist = data['mol2_bondlist']
        inst.empty = False
        return inst

    @classmethod
    def fcc(cls, a: float, nc=[1,1,1]):
        """Generates a Coordinates object that represents an FCC crystal.

        Args:
            a (float): lattice parameter
            nc (list): number of unit cells in the three lattice vector directions, defaults to [1,1,1]

        Returns:
            Coordinates: a Coordinates object
        """
        from htpolynet.geometry.lattice import fcc as _fcc
        posn = _fcc(a,nc)
        N = len(posn)
        inst = cls()
        inst.A['globalIdx'] = list(range(1,N+1))
        inst.A['atomName'] = 'AL'
        inst.A['resNum'] = list(range(1,N+1))
        inst.A['posX'] = posn[:,0]
        inst.A['posY'] = posn[:,1]
        inst.A['posZ'] = posn[:,2]
        inst.A['resName'] = 'MET'
        inst.N = N
        return inst

    def claim_parent(self, parent):
        self.parent = parent

    def set_box(self, box: np.ndarray):
        """Sets the box size from box.

        Args:
            box (numpy.ndarray): 3-by-1 or 3-by-3 box size matrix
        """
        if box.shape == (3, 1):
            for i in range(3):
                self.box[i, i] = box[i]
        elif box.shape == (3, 3):
            self.box = np.copy(box)

    def total_volume(self, units='gromacs'):
        """Returns total volume of box.

        Args:
            units (str): unit system designation; if 'SI' returns m^3, defaults to 'gromacs'

        Returns:
            float: volume (in nm^3 if units is 'gromacs' or m^3 if units is 'SI')
        """
        nm_per_m = 1.e9
        vol = np.prod(self.box.diagonal())  # nm^3
        return vol if units != 'SI' else vol / (nm_per_m ** 3)

    def copy_coords(self, other):
        """Copies the posX, posY, and posZ atom attributes, and the box size, from other.A to self.A.

        Args:
            other (Coordinates): the other Coordinates instance
        """
        assert self.A.shape[0] == other.A.shape[0], f'Cannot copy -- atom count mismatch {self.A.shape[0]} vs {other.A.shape[0]}'
        for c in ['posX', 'posY', 'posZ']:
            otherpos = other.A[c].copy()
            self.A[c] = otherpos
        self.box = np.copy(other.box)

    def subcoords(self, sub_adf: pd.DataFrame):
        """Generates a new Coordinates object to hold the atoms dataframe in 'sub_adf' parameter.

        Args:
            sub_adf (pd.DataFrame): an atom dataframe

        Returns:
            Coordinates: a new Coordinates object
        """
        newC = Coordinates()
        newC.set_box(self.box)
        newC.A = sub_adf
        newC.N = sub_adf.shape[0]
        return newC
    
    def unwrap(self, P, O, pbc):
        """Shifts point P to its unwrapped closest periodic image to point O.

        Args:
            P (np.ndarray(3,float)): a point
            O (np.ndarray(3,float)): origin
            pbc (np.ndarray(3,int)): directions in which pbc are applied

        Returns:
            np.ndarray(3,float): a point
        """
        ROP = self.mic(O - P, pbc)
        PCPI = O - ROP
        return PCPI

    def pierces(self, B: pd.DataFrame, C: pd.DataFrame, pbc):
        """Determines whether or not bond represented by two points in B pierces ring represented by N points in C.

        Args:
            B (pd.DataFrame): two points defining a bond
            C (pd.DataFrame): N points defining a ring
            pbc (list): periodic boundary condition flags, one per dimension

        Returns:
            bool: True if ring is pierced
        """
        BC = np.array(B[['posX', 'posY', 'posZ']])
        CC = np.array(C[['posX', 'posY', 'posZ']])
        # get both atoms in bond into CPI
        # get all atoms in ring into CPI (but not necessarily wrt bond)
        CC[1:] = np.array([self.unwrap(c, CC[0], pbc) for c in CC[1:]])
        BC[0] = self.unwrap(BC[0], CC[0], pbc)
        BC[1] = self.unwrap(BC[1], BC[0], pbc)
        B[['posX', 'posY', 'posZ']] = BC
        C[['posX', 'posY', 'posZ']] = CC
        S = Segment(BC)
        R = Ring(CC)
        R.analyze()
        do_it, point = R.segint(S)
        return do_it

    def linkcell_initialize(self, cutoff=0.0, populate=True, force_repopulate=False, save=True):
        """Initializes the link-cell structure for ring-pierce testing.

        Assigns a ``linkcell_idx`` attribute to each ring atom and reactive
        atom (z > 0) directly in ``self.A``; all other atoms get -1.
        Results are cached to a ``.grx`` file keyed by cutoff value.

        Args:
            cutoff (float): cell side length (nm), defaults to 0.0
            populate (bool): if True, perform the assignment; defaults to True
            force_repopulate (bool): if True, ignore any cached file; defaults to False
            save (bool): if True, write the linkcell_idx column to a cache file; defaults to True
        """
        logger.debug('Initializing link-cell structure')
        self.linkcell.create(cutoff, self.box)
        if populate:
            lc_file = f'linkcell-{cutoff:.2f}.grx'
            if os.path.exists(lc_file) and not force_repopulate:
                logger.debug(f'Found {lc_file}; reading cached linkcell_idx.')
                self.read_atomset_attributes(lc_file)
            else:
                mask = (
                    self.A['globalIdx'].isin(self.parent.Topology.rings.all_atoms())
                    | (self.A['z'] > 0)
                )
                self.linkcell.assign(self.A, mask=mask)
                if save:
                    self.write_atomset_attributes(['linkcell_idx'], lc_file)

    def geometric_center(self):
        """Computes and returns the geometric center of the atoms in self's A dataframe.

        Returns:
            np.ndarray(3,float): geometric center
        """
        a = self.A
        return np.array([a.posX.mean(), a.posY.mean(), a.posZ.mean()])

    def rij(self, i, j, pbc=None):
        """Computes distance between atoms i and j.

        Args:
            i (int): globalIdx of first atom
            j (int): globalIdx of second atom
            pbc (array-like, optional): PBC flags per axis, defaults to [1,1,1]

        Returns:
            float: distance between i and j (nm)
        """
        if pbc is None:
            pbc = [1, 1, 1]
        if np.any(pbc) and np.allclose(self.box, 0):
            logger.warning('Interatomic distance calculation using PBC with no boxsize set.')
        Rij = self.mic(self.get_R(i) - self.get_R(j), pbc)
        return np.sqrt(Rij.dot(Rij))

    def mic(self, r, pbc=None):
        """Applies minimum image convention to displacement vector r.

        Args:
            r (np.ndarray(3,float)): displacement vector
            pbc (array-like, optional): PBC flags per axis, defaults to [1,1,1]

        Returns:
            np.ndarray(3,float): minimum-image displacement vector
        """
        if pbc is None:
            pbc = [1, 1, 1]
        b = self.box.diagonal()
        mask = np.asarray(pbc, dtype=bool)
        r = r.copy()
        r[mask] -= np.round(r[mask] / b[mask]) * b[mask]
        return r

    def wrap_point(self, ri):
        """Wraps point ri into the central periodic image.

        Args:
            ri (np.ndarray(3,float)): a point

        Returns:
            tuple: the wrapped point and number of box lengths shifted per dimension
        """
        b = self.box.diagonal()
        R = ri % b
        box_lengths = (-np.floor(ri / b)).astype(int)
        return R, box_lengths

    def wrap_coords(self):
        """Wraps all atomic coordinates into box."""
        assert np.any(self.box), f'Cannot wrap if boxsize is not set: {self.box}'
        b = self.box.diagonal()
        pos = self.A[['posX','posY','posZ']].values
        box_lengths = (-np.floor(pos / b)).astype(int)
        self.A[['posX','posY','posZ']] = pos % b
        self.A[['boxLx','boxLy','boxLz']] = box_lengths

    def merge(self, other):
        """Merges two Coordinates objects.

        Args:
            other (Coordinates): the other Coordinates object

        Returns:
            tuple: integer shifts in atom index, bond index, and residue index
        """
        idxshift = self.A.shape[0]
        bdxshift = self.mol2_bonds.shape[0]
        rdxshift = 0 if self.A.empty else int(self.A['resNum'].max())
        nOtherBonds = 0
        if not other.A.empty:
            oa = other.A.copy()
            oa['globalIdx'] += idxshift
            oa['resNum'] += rdxshift
            self.A = pd.concat((self.A, oa), ignore_index=True)
            self.N = self.A.shape[0]
        if not other.mol2_bonds.empty:
            ob = other.mol2_bonds.copy()
            nOtherBonds = ob.shape[0]
            ob['bondIdx'] += bdxshift
            ob[['ai', 'aj']] += idxshift
            self.mol2_bonds = pd.concat((self.mol2_bonds, ob), ignore_index=True)
            self.mol2_bondlist.update(ob)
        self.metadat['N'] = self.N
        self.metadat['nBonds'] = self.metadat.get('nBonds', 0) + nOtherBonds
        return (idxshift, bdxshift, rdxshift)
            
    def write_atomset_attributes(self, attributes, filename, formatters=[]):
        """Writes atom attributes to a GRX-format file.

        Args:
            attributes (list): list of attribute names to write
            filename (str): name of file to write
            formatters (list): formatting methods per attribute, defaults to []

        Raises:
            Exception: if any item in attributes does not exist in the coordinates dataframe
        """
        _gro_io.write_grx(filename, self.A, attributes, formatters=formatters)

    def read_atomset_attributes(self, filename, attributes=[]):
        """Reads atomic attributes from a GRX-format file into self.A.

        Args:
            filename (str): name of file
            attributes (list): list of attributes to take, defaults to [] (take all)

        Returns:
            list: names of attributes that were read
        """
        df, attributes_read = _gro_io.read_grx(filename, attributes)
        self.A = self.A.merge(df, how='outer', on='globalIdx')
        return attributes_read

    def set_atomset_attribute(self, attribute, srs):
        """Sets attribute of atoms to srs.

        Args:
            attribute (str): name of attribute
            srs (scalar or list-like): attribute values in same ordering as self.A
        """
        self.A[attribute] = srs

    def atomcount(self):
        """Returns the number of atoms in the Coordinates object.

        Returns:
            int: number of atoms
        """
        return self.N

    def show_z_report(self):
        """Generates a text-based histogram of atom z-values (0-3), keyed by resname:atomname."""
        zhists = {}
        for r in self.A.itertuples():
            n = r.atomName
            nn = r.resName
            k = f'{nn}:{n}'
            z = r.z
            if not k in zhists:
                zhists[k] = np.zeros(4).astype(int)
            zhists[k][z] += 1
        for n in zhists:
            if any([zhists[n][i] > 0 for i in range(1, 4)]):
                logger.debug(f'Z-hist for {n} atoms:')
                for i in range(4):
                    logger.debug(f'{i:>5d} ({zhists[n][i]:>6d}): ' + '*' * (zhists[n][i] // 10))

    def return_bond_lengths(self, bdf):
        """Returns an ordered list of bond lengths for bonds in bdf.

        Args:
            bdf (pd.DataFrame): dataframe with 'ai' and 'aj' columns of atom indices indicating bonds

        Returns:
            list: atom distances
        """
        lengths = []
        for b in bdf.itertuples():
            lengths.append(self.rij(b.ai, b.aj))
        return lengths

    def minimum_distance(self, other, self_excludes=None, other_excludes=None):
        """Computes and returns distance of closest approach between two sets of atoms.

        Args:
            other (Coordinates): other Coordinates instance
            self_excludes (list, optional): atom globalIdx values in self to exclude, defaults to None
            other_excludes (list, optional): atom globalIdx values in other to exclude, defaults to None

        Returns:
            float: minimum interatomic distance (nm)
        """
        sp = self.A.loc[~self.A['globalIdx'].isin(self_excludes or []), ['posX','posY','posZ']].values
        op = other.A.loc[~other.A['globalIdx'].isin(other_excludes or []), ['posX','posY','posZ']].values
        diff = sp[:, np.newaxis, :] - op[np.newaxis, :, :]
        return np.sqrt((diff * diff).sum(axis=2)).min()
    
    def homog_trans(self, M: Matrix4, indices=None):
        """Applies homogeneous transformation matrix M to coordinates.

        Args:
            M (Matrix4): homogeneous transformation matrix
            indices (list, optional): atom globalIdx values to transform; if None transforms all
        """
        R = M.m[:3, :3]
        t = M.m[:3, 3]
        if not indices:
            pos = self.A[['posX','posY','posZ']].values
            self.A[['posX','posY','posZ']] = pos @ R.T + t
        else:
            mask = self.A['globalIdx'].isin(indices)
            pos = self.A.loc[mask, ['posX','posY','posZ']].values
            self.A.loc[mask, ['posX','posY','posZ']] = pos @ R.T + t

    def rotate(self, R):
        """Rotates all coordinate vectors by rotation matrix R.

        Args:
            R (numpy.ndarray): rotation matrix (3x3)
        """
        pos = self.A[['posX','posY','posZ']].values
        self.A[['posX','posY','posZ']] = pos @ R.T

    def translate(self, L):
        """Translates all coordinate vectors by displacement vector L.

        Args:
            L (numpy.ndarray): displacement vector (nm)
        """
        pos = self.A[['posX','posY','posZ']].values
        self.A[['posX','posY','posZ']] = pos + L

    def maxspan(self):
        """Returns dimensions of orthorhombic convex hull enclosing Coordinates.

        Returns:
            numpy.ndarray: array of x-span, y-span, z-span
        """
        sp = self.A[['posX', 'posY', 'posZ']]
        return np.array(
            [
                sp.posX.max() - sp.posX.min(),
                sp.posY.max() - sp.posY.min(),
                sp.posZ.max() - sp.posZ.min()
            ]
        )

    def minmax(self):
        """Returns the lower-leftmost and upper-rightmost positions in the atoms dataframe.

        Returns:
            tuple(np.ndarray, np.ndarray): lower-leftmost and upper-rightmost points, respectively
        """
        sp = self.A[['posX', 'posY', 'posZ']]
        return np.array([sp.posX.min(), sp.posY.min(), sp.posZ.min()]), np.array([sp.posX.max(), sp.posY.max(), sp.posZ.max()])

    def checkbox(self):
        """Checks that all atom positions fit within the designated box.

        Returns:
            tuple(bool, bool): True,True if both lower-leftmost and upper-rightmost points are within the box
        """
        mm, MM = self.minmax()
        bb = self.box.diagonal()
        return mm < bb, MM > bb

    def get_R(self, idx: int) -> np.ndarray:
        """Returns the cartesian position of atom with global index idx.

        Args:
            idx (int): global index of atom

        Returns:
            numpy.ndarray: cartesian position of the atom, shape (3,)
        """
        return self.A.loc[self.A['globalIdx'] == idx, ['posX', 'posY', 'posZ']].values[0]
    
    def get_atom_attribute(self, name, attributes):
        """Returns values of attributes listed in name from atoms specified by attribute:value pairs.

        Args:
            name (str or list): attribute(s) whose values are to be returned
            attributes (dict): attribute:value pairs that specify the set of atoms to be considered

        Returns:
            scalar or Series: attribute value(s) of the matching row
        """
        mask = pd.Series(True, index=self.A.index)
        for k, v in attributes.items():
            mask &= (self.A[k] == v)
        return self.A.loc[mask, name].iloc[0]
    
    def get_atoms_w_attribute(self,name,attributes):
        """Returns rows of atoms dataframe with columns named in name for atoms matching attributes.

        Args:
            name (list): attribute names for the columns to be returned
            attributes (dict): attribute:value pairs that specify the set of atoms to be considered

        Returns:
            pd.DataFrame: matching dataframe segment
        """
        df = self.A
        return get_rows_w_attribute(df, name, attributes)

    def set_atom_attribute(self, name, value, attributes):
        """Sets the attributes named in name to the given values for atoms matching attributes.

        Args:
            name (list): names of attributes to set
            value (list): values of attributes to set (parallel to name)
            attributes (dict): attribute:value pairs that specify the set of atoms to be considered
        """
        df = self.A
        set_row_attribute(df, name, value, attributes)

    def delete_atoms(self, idx=[], reindex=True):
        """Deletes atoms whose global indices appear in idx.

        If reindex is True, global indices are recalculated to be sequential starting at 1
        with no gaps, and two new columns are added: 'oldGlobalIdx' (pre-deletion indices)
        and 'globalIdxShift' (change from old to new index for each atom).

        Args:
            idx (list): atom indexes to delete, defaults to []
            reindex (bool): reindex remaining atoms, defaults to True
        """
        # logger.debug(f'Coordinates:delete_atoms {idx}')
        adf = self.A
        indexes_to_drop = adf[adf.globalIdx.isin(idx)].index
        indexes_to_keep = set(range(adf.shape[0])) - set(indexes_to_drop)
        self.A = adf.take(list(indexes_to_keep)).reset_index(drop=True)
        if reindex:
            adf = self.A
            oldGI = adf['globalIdx'].copy()
            adf['globalIdx'] = adf.index + 1
            mapper = {k: v for k, v in zip(oldGI, adf['globalIdx'])}
        self.N -= len(idx)
        ''' delete appropriate bonds '''
        if not self.mol2_bonds.empty:
            d = self.mol2_bonds
            indexes_to_drop = d[(d.ai.isin(idx)) | (d.aj.isin(idx))].index
            indexes_to_keep = set(range(d.shape[0])) - set(indexes_to_drop)
            self.mol2_bonds = d.take(list(indexes_to_keep)).reset_index(drop=True)
            if reindex:
                d = self.mol2_bonds
                d.ai = d.ai.map(mapper)
                d.aj = d.aj.map(mapper)
                d.bondIdx = d.index + 1
            if 'nBonds' in self.metadat:
                self.metadat['nBonds'] = len(self.mol2_bonds)
            self.bondlist = Bondlist.fromDataFrame(self.mol2_bonds)

    def write_gro(self, filename, grotitle=''):
        """Writes coordinates and, if present, velocities to a GROMACS-format coordinate file.

        Args:
            filename (str): name of file to write
            grotitle (str): title line for the .gro file, defaults to ''
        """
        _gro_io.write(filename, self.A, self.box, self.name, grotitle=grotitle)

    def write_mol2(self, filename, bondsDF=pd.DataFrame(), molname='', other_attributes=pd.DataFrame()):
        """Writes a mol2-format file from coordinates and an optional bonds DataFrame.

        Args:
            filename (str): name of file to write
            bondsDF (pandas.DataFrame): dataframe of bonds ['ai','aj'], defaults to empty DataFrame
            molname (str): name of molecule, defaults to ''
            other_attributes (pandas.DataFrame): auxiliary dataframe of attributes, defaults to empty DataFrame
        """
        _mol2_io.write(
            filename, self.A, self.mol2_bonds, self.metadat, self.name,
            bondsDF=bondsDF, molname=molname, other_attributes=other_attributes,
        )
