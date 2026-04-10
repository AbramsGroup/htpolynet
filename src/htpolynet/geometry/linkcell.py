"""Manages the link-cell structure used for ring-pierce testing.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import numpy as np
import logging
from itertools import product

logger = logging.getLogger(__name__)


class Linkcell:
    """Spatial hash grid for efficient ring-pierce candidate pruning.

    Workflow:
        1. ``create(cutoff, box)``  — build the grid geometry and precompute
           per-cell neighbour lists.
        2. ``assign(A, mask)``      — vectorised, in-place assignment of
           ``linkcell_idx`` to every atom row in *A* that satisfies *mask*;
           all other rows get ``linkcell_idx = -1``.
        3. ``neighbor_cell_set(ci)`` — returns the set of cell indices
           (including *ci* itself) that are spatially adjacent to cell *ci*.
        4. ``nearby_atom_ids(A, cell_set)`` — vectorised ``isin`` query;
           returns the ``globalIdx`` values of atoms assigned to any cell in
           *cell_set*.

    No memberlists are maintained; all proximity queries operate directly on
    the ``linkcell_idx`` column of the atom DataFrame.
    """

    def __init__(self, box=[], cutoff=None, pbc_wrapper=None):
        self.box = box
        self.cutoff = cutoff
        self.pbc_wrapper = pbc_wrapper

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def create(self, cutoff, box, origin=np.array([0., 0., 0.])):
        """Build the link-cell grid and precompute per-cell neighbour lists.

        Args:
            cutoff (float): minimum cell side length (nm)
            box (numpy.ndarray): simulation box; shape (3,) or (3,3)
            origin (numpy.ndarray): grid origin, defaults to [0,0,0]
        """
        if box.shape == (3, 3):
            box = np.diagonal(box)
        self.cutoff = cutoff
        self.box = box
        self.origin = origin
        self.ncells = np.floor(self.box / self.cutoff).astype(int)
        self.celldim = box / self.ncells
        self.cells = np.zeros((*self.ncells, 3))
        self.cellndx = np.array(list(product(*[np.arange(x) for x in self.ncells])))
        for t in self.cellndx:
            i, j, k = t
            self.cells[i, j, k] = self.celldim * np.array([i, j, k]) + self.origin
        self.make_neighborlists()
        logger.debug(
            f'Linkcell structure: {len(self.cellndx)} cells ({self.ncells}) '
            f'dim {self.celldim}'
        )

    def make_neighborlists(self):
        """Precompute per-cell neighbour lists (26 neighbours each, with PBC)."""
        n = self.cellndx.shape[0]
        self.neighborlists = [[] for _ in range(n)]
        for C in self.cellndx:
            idx = self.ldx_of_cellndx(C)
            for D in self.neighbors_of_cellndx(C):
                d_idx = self.ldx_of_cellndx(D)
                if d_idx != idx:
                    self.neighborlists[idx].append(d_idx)

    # ------------------------------------------------------------------
    # Index assignment (replaces populate / populate_par / make_memberlists)
    # ------------------------------------------------------------------

    def assign(self, A, mask=None):
        """Assign ``linkcell_idx`` to atoms in *A* in-place, no copies.

        All atoms start with ``linkcell_idx = -1``.  Those selected by
        *mask* (a boolean Series aligned with *A*) are assigned the correct
        cell index via a single vectorised numpy operation.

        Args:
            A (pd.DataFrame): atom data frame; must contain posX/posY/posZ
            mask (pd.Series of bool, optional): selects atoms to assign;
                if None, all atoms are assigned
        """
        A['linkcell_idx'] = -1
        rows = A.index if mask is None else A.index[mask]
        if len(rows) == 0:
            return
        pos = A.loc[rows, ['posX', 'posY', 'posZ']].values
        C = np.floor(pos * np.reciprocal(self.celldim)).astype(int)
        C = np.clip(C, 0, self.ncells - 1)
        nc = self.ncells
        A.loc[rows, 'linkcell_idx'] = (
            C[:, 0] * nc[1] * nc[2] + C[:, 1] * nc[2] + C[:, 2]
        )
        n_assigned = len(rows)
        ldx = A.loc[rows, 'linkcell_idx'].values
        counts = np.bincount(ldx, minlength=self.cellndx.shape[0])
        logger.debug(
            f'Linkcell: assigned {n_assigned} atoms; '
            f'avg/min/max cell pop: '
            f'{counts[counts>0].mean():.3f}/'
            f'{int(counts.min())}/'
            f'{int(counts.max())}'
        )

    # ------------------------------------------------------------------
    # Proximity queries
    # ------------------------------------------------------------------

    def neighbor_cell_set(self, cell_idx):
        """Return the set of cell indices adjacent to *cell_idx*, including itself.

        Args:
            cell_idx (int): scalar cell index

        Returns:
            set: neighbour cell indices plus *cell_idx*
        """
        return set(self.neighborlists[cell_idx]) | {cell_idx}

    def nearby_atom_ids(self, A, cell_set):
        """Return ``globalIdx`` values of atoms whose cell is in *cell_set*.

        Args:
            A (pd.DataFrame): atom data frame with ``linkcell_idx`` column
            cell_set (set): cell indices to search

        Returns:
            numpy.ndarray: globalIdx values
        """
        return A.loc[A['linkcell_idx'].isin(cell_set), 'globalIdx'].values

    # ------------------------------------------------------------------
    # Low-level geometry helpers (kept for utility use)
    # ------------------------------------------------------------------

    def ldx_of_cellndx(self, C):
        """Return scalar index of cell with (i,j,k)-index C."""
        nc = self.ncells
        return int(C[0] * nc[1] * nc[2] + C[1] * nc[2] + C[2])

    def cellndx_of_ldx(self, i):
        """Return (i,j,k)-index of cell with scalar index i."""
        return self.cellndx[i]

    def cellndx_in_structure(self, C):
        """Return True if (i,j,k)-index C is within the grid."""
        return all(np.zeros(3) <= C) and all(C < self.ncells)

    def neighbors_of_cellndx(self, Ci):
        """Return the 26 (i,j,k)-index neighbours of cell Ci, with PBC wrapping."""
        assert self.cellndx_in_structure(Ci), \
            f'Error: cell {Ci} outside of cell structure {self.ncells}'
        retlist = []
        dd = np.array([-1, 0, 1])
        for s in product(dd, dd, dd):
            if s == (0, 0, 0):
                continue
            nCi = Ci + np.array(s)
            for d in range(3):
                if nCi[d] == self.ncells[d]:
                    nCi[d] = 0
                elif nCi[d] == -1:
                    nCi[d] = self.ncells[d] - 1
            retlist.append(nCi)
        return retlist

    def cellndx_of_point(self, R):
        """Return the (i,j,k) cell index of point R (uses pbc_wrapper)."""
        wrapR, bl = self.pbc_wrapper(R)
        C = np.floor(wrapR * np.reciprocal(self.celldim)).astype(int)
        lowdim = (C < np.zeros(3).astype(int)).astype(int)
        hidim = (C >= self.ncells).astype(int)
        if any(lowdim) or any(hidim):
            logger.warning(f'Warning: point {R} maps to out-of-bounds-cell {C} ({self.ncells})')
        C += lowdim
        C -= hidim
        return C

    def point_in_cellndx(self, R, C):
        """Return True if point R is located in cell C."""
        LL, UU = self.corners_of_cellndx(C)
        wrapR, bl = self.pbc_wrapper(R)
        return all(wrapR < UU) and all(wrapR >= LL)

    def corners_of_cellndx(self, C):
        """Return lower-left and upper-right corners of cell C."""
        LL = self.cells[C[0], C[1], C[2]]
        return np.array([LL, LL + self.celldim])

    def are_cellndx_neighbors(self, Ci, Cj):
        """Return True if cells Ci and Cj are adjacent (with PBC)."""
        assert self.cellndx_in_structure(Ci)
        assert self.cellndx_in_structure(Cj)
        dij = Ci - Cj
        low = (dij < -self.ncells / 2).astype(int)
        hi = (dij > self.ncells / 2).astype(int)
        dij += (low - hi) * self.ncells
        return all(x in [-1, 0, 1] for x in dij)
