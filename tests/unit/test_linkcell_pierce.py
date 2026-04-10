"""Unit tests for the refactored Linkcell and the ring-pierce pipeline.

Tests cover:
  1. Linkcell.create() / assign() / neighbor_cell_set() / nearby_atom_ids()
  2. RingList.filter() with atom globalIdx (not cell indices)
  3. Full assign → neighbor_cell_set → nearby_atom_ids → filter → pierced_by chain

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import unittest
import numpy as np
import pandas as pd
import networkx as nx

from htpolynet.geometry.linkcell import Linkcell
from htpolynet.geometry.ring import Ring, RingList


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atom_df(positions, start_idx=1):
    """Build a minimal atom DataFrame from an (N,3) position array."""
    N = len(positions)
    return pd.DataFrame({
        'globalIdx': list(range(start_idx, start_idx + N)),
        'posX': positions[:, 0],
        'posY': positions[:, 1],
        'posZ': positions[:, 2],
        'linkcell_idx': -1,
    })


def _hexagon_xy(radius=0.5, center=(0.0, 0.0, 0.0), start_idx=1):
    """Six-atom regular hexagon lying in z=0 plane."""
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    cx, cy, cz = center
    pos = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
        np.full(6, cz),
    ])
    return _make_atom_df(pos, start_idx=start_idx)


# ---------------------------------------------------------------------------
# Linkcell unit tests
# ---------------------------------------------------------------------------

class TestLinkcellCreate(unittest.TestCase):

    def setUp(self):
        self.lc = Linkcell()
        self.box = np.array([10.0, 10.0, 10.0])
        self.cutoff = 2.0  # → 5×5×5 = 125 cells

    def test_ncells(self):
        self.lc.create(self.cutoff, self.box)
        np.testing.assert_array_equal(self.lc.ncells, [5, 5, 5])

    def test_celldim(self):
        self.lc.create(self.cutoff, self.box)
        np.testing.assert_array_almost_equal(self.lc.celldim, [2.0, 2.0, 2.0])

    def test_neighborlist_length(self):
        self.lc.create(self.cutoff, self.box)
        # 5×5×5 grid: every cell has exactly 26 neighbours (full PBC)
        for nl in self.lc.neighborlists:
            self.assertEqual(len(nl), 26)

    def test_create_3x3_box(self):
        """create() accepts (3,3) box and extracts diagonal."""
        box3 = np.diag(self.box)
        self.lc.create(self.cutoff, box3)
        np.testing.assert_array_equal(self.lc.ncells, [5, 5, 5])


class TestLinkcellAssign(unittest.TestCase):

    def setUp(self):
        self.lc = Linkcell()
        self.box = np.array([10.0, 10.0, 10.0])
        self.lc.create(2.0, self.box)  # 5×5×5 cells, celldim = 2 nm

    def _atom_at(self, x, y, z, gidx=1):
        return _make_atom_df(np.array([[x, y, z]]), start_idx=gidx)

    def test_assign_sets_column(self):
        A = _make_atom_df(np.random.uniform(0, 10, (20, 3)))
        self.lc.assign(A)
        self.assertTrue((A['linkcell_idx'] >= 0).all())

    def test_all_indices_in_range(self):
        A = _make_atom_df(np.random.uniform(0, 10, (50, 3)))
        self.lc.assign(A)
        n_total = int(np.prod(self.lc.ncells))
        self.assertTrue((A['linkcell_idx'] < n_total).all())

    def test_known_position_cell_index(self):
        """Atom at (1,1,1) should land in cell (0,0,0) → index 0."""
        A = self._atom_at(1.0, 1.0, 1.0)
        self.lc.assign(A)
        # cell (0,0,0) has scalar index 0
        self.assertEqual(int(A.iloc[0]['linkcell_idx']), 0)

    def test_known_position_cell_index_2(self):
        """Atom at (3,1,1) should land in cell (1,0,0) → index nc[1]*nc[2] = 25."""
        A = self._atom_at(3.0, 1.0, 1.0)
        self.lc.assign(A)
        expected = 1 * 5 * 5  # i*nc[1]*nc[2]
        self.assertEqual(int(A.iloc[0]['linkcell_idx']), expected)

    def test_mask_unmasked_stay_minus_one(self):
        """Atoms excluded by mask must keep linkcell_idx = -1."""
        A = _make_atom_df(np.random.uniform(0, 10, (10, 3)))
        mask = pd.Series([True] * 5 + [False] * 5, index=A.index)
        self.lc.assign(A, mask=mask)
        self.assertTrue((A.loc[mask, 'linkcell_idx'] >= 0).all())
        self.assertTrue((A.loc[~mask, 'linkcell_idx'] == -1).all())

    def test_empty_mask_no_assignment(self):
        A = _make_atom_df(np.random.uniform(0, 10, (5, 3)))
        mask = pd.Series([False] * 5, index=A.index)
        self.lc.assign(A, mask=mask)
        self.assertTrue((A['linkcell_idx'] == -1).all())

    def test_inplace_modifies_original(self):
        """assign() must mutate the caller's DataFrame, not a copy."""
        A = _make_atom_df(np.random.uniform(0, 10, (10, 3)))
        before_id = id(A)
        self.lc.assign(A)
        self.assertEqual(id(A), before_id)
        self.assertTrue((A['linkcell_idx'] >= 0).all())


class TestLinkcellNeighborCellSet(unittest.TestCase):

    def setUp(self):
        self.lc = Linkcell()
        self.lc.create(2.0, np.array([10.0, 10.0, 10.0]))  # 5×5×5

    def test_includes_self(self):
        for ci in [0, 12, 62, 124]:
            s = self.lc.neighbor_cell_set(ci)
            self.assertIn(ci, s)

    def test_size_is_27(self):
        """A cell in a 5×5×5 PBC grid has 27 cells in its neighbourhood (self + 26)."""
        for ci in [0, 7, 63, 124]:
            s = self.lc.neighbor_cell_set(ci)
            self.assertEqual(len(s), 27)

    def test_symmetry(self):
        """If B is in A's neighbourhood, A is in B's neighbourhood."""
        for ci in [0, 13, 62]:
            for cj in self.lc.neighbor_cell_set(ci):
                self.assertIn(ci, self.lc.neighbor_cell_set(cj))


class TestLinkcellNearbyAtomIds(unittest.TestCase):

    def setUp(self):
        self.lc = Linkcell()
        self.lc.create(2.0, np.array([10.0, 10.0, 10.0]))

    def test_returns_atoms_in_cells(self):
        A = _make_atom_df(np.array([
            [1.0, 1.0, 1.0],  # cell (0,0,0) → idx 0
            [3.0, 1.0, 1.0],  # cell (1,0,0) → idx 25
            [7.0, 7.0, 7.0],  # cell (3,3,3) → idx 93
        ]))
        self.lc.assign(A)
        cell_set = {0}
        ids = self.lc.nearby_atom_ids(A, cell_set)
        self.assertIn(1, ids)
        self.assertNotIn(2, ids)
        self.assertNotIn(3, ids)

    def test_empty_cell_set(self):
        A = _make_atom_df(np.random.uniform(0, 10, (10, 3)))
        self.lc.assign(A)
        ids = self.lc.nearby_atom_ids(A, set())
        self.assertEqual(len(ids), 0)

    def test_unassigned_atoms_excluded(self):
        """Atoms with linkcell_idx == -1 must never appear (no cell has index -1)."""
        A = _make_atom_df(np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]))
        mask = pd.Series([True, False], index=A.index)
        self.lc.assign(A, mask=mask)
        all_cells = set(range(int(np.prod(self.lc.ncells))))
        ids = self.lc.nearby_atom_ids(A, all_cells)
        # only globalIdx=1 was assigned
        self.assertIn(1, ids)
        self.assertNotIn(2, ids)


# ---------------------------------------------------------------------------
# RingList.filter — atom globalIdx (not cell indices)
# ---------------------------------------------------------------------------

class TestRingListFilter(unittest.TestCase):
    """Regression test for the bug where cell indices were passed to filter()."""

    def _make_ringlist(self):
        # Ring A: atoms 1-6; Ring B: atoms 11-16
        rl = RingList([
            Ring([1, 2, 3, 4, 5, 6]),
            Ring([11, 12, 13, 14, 15, 16]),
        ])
        return rl

    def test_filter_returns_matching_ring(self):
        rl = self._make_ringlist()
        result = rl.filter([3, 7, 20])  # 3 is in ring A
        self.assertEqual(len(result), 1)
        self.assertIn(Ring([1, 2, 3, 4, 5, 6]), result)

    def test_filter_returns_both_rings(self):
        rl = self._make_ringlist()
        result = rl.filter([5, 14])  # one atom in each ring
        self.assertEqual(len(result), 2)

    def test_filter_returns_empty_for_no_match(self):
        rl = self._make_ringlist()
        result = rl.filter([100, 200, 300])
        self.assertEqual(len(result), 0)

    def test_filter_cell_indices_do_not_match_atom_ids(self):
        """Cell index 0 must NOT match ring atoms [1..6] — guard against the old bug."""
        rl = self._make_ringlist()
        # Small cell indices (0, 1, 2) that happen to not be atom globalIdx
        result = rl.filter([0])
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# Full pipeline: assign → neighbor_cell_set → nearby_atom_ids → filter → pierced_by
# ---------------------------------------------------------------------------

class TestRingPiercePipeline(unittest.TestCase):
    """Integration test for the complete ring-pierce chain.

    The system: a regular hexagon centred at the origin in the z=0 plane,
    inscribed in a 10 nm box.  A bond running from (0,0,−1) to (0,0,+1)
    through the centre should pierce the ring; a bond offset to (5,5,±1)
    should not.
    """

    RADIUS = 0.5   # nm — hexagon radius
    BOX    = 10.0  # nm
    CUT    = 1.0   # nm — linkcell cutoff

    def setUp(self):
        # ---- atoms: hexagon at box centre --------------------------------
        cx, cy, cz = 5.0, 5.0, 5.0
        self.ring_df = _hexagon_xy(
            radius=self.RADIUS,
            center=(cx, cy, cz),
            start_idx=1,
        )
        # two dummy "bond-endpoint" atoms far from the ring
        extra = _make_atom_df(np.array([[cx, cy, cz - 1.0],
                                        [cx, cy, cz + 1.0]]), start_idx=7)
        self.A = pd.concat([self.ring_df, extra], ignore_index=True)

        # ---- linkcell ----------------------------------------------------
        self.lc = Linkcell()
        self.lc.create(self.CUT, np.array([self.BOX, self.BOX, self.BOX]))
        self.lc.assign(self.A)

        # ---- ring list with coordinates ---------------------------------
        self.rings = RingList([Ring(list(range(1, 7)))])
        self.rings.injest_coordinates(self.A)

    def _nearby_rings_for_bond(self, gidx_i, gidx_j):
        """Run the full linkcell → filter pipeline for bond (i,j)."""
        ci = int(self.A.loc[self.A['globalIdx'] == gidx_i, 'linkcell_idx'].iloc[0])
        cj = int(self.A.loc[self.A['globalIdx'] == gidx_j, 'linkcell_idx'].iloc[0])
        cell_set = self.lc.neighbor_cell_set(ci) | self.lc.neighbor_cell_set(cj)
        atom_ids = self.lc.nearby_atom_ids(self.A, cell_set)
        return self.rings.filter(atom_ids)

    def test_piercing_bond_found_nearby(self):
        """Bond endpoints in cells adjacent to the ring must see the ring."""
        # atoms 7 (below) and 8 (above) straddle the hexagon centre
        nearby = self._nearby_rings_for_bond(7, 8)
        self.assertGreater(len(nearby), 0, "ring should appear in nearby set")

    def test_piercing_bond_actually_pierces(self):
        cx, cy, cz = 5.0, 5.0, 5.0
        B = np.array([[cx, cy, cz - 1.0],
                      [cx, cy, cz + 1.0]])
        did, _ = self.rings[0].pierced_by(B)
        self.assertTrue(did)

    def test_non_piercing_bond_not_nearby(self):
        """A bond far from the ring (corner of box) should not reach the ring."""
        far = _make_atom_df(np.array([[0.5, 0.5, 0.5],
                                      [0.5, 0.5, 1.5]]), start_idx=9)
        A2 = pd.concat([self.A, far], ignore_index=True)
        self.lc.assign(A2)  # re-assign with new atoms
        ci = int(A2.loc[A2['globalIdx'] == 9, 'linkcell_idx'].iloc[0])
        cj = int(A2.loc[A2['globalIdx'] == 10, 'linkcell_idx'].iloc[0])
        cell_set = self.lc.neighbor_cell_set(ci) | self.lc.neighbor_cell_set(cj)
        atom_ids = self.lc.nearby_atom_ids(A2, cell_set)
        nearby = self.rings.filter(atom_ids)
        self.assertEqual(len(nearby), 0, "far bond should not see the ring")

    def test_non_piercing_bond_does_not_pierce(self):
        """Bond passing through the ring plane but outside the hexagon."""
        cx, cy, cz = 5.0, 5.0, 5.0
        # offset by 5× the ring radius — clearly outside
        offset = self.RADIUS * 5
        B = np.array([[cx + offset, cy + offset, cz - 1.0],
                      [cx + offset, cy + offset, cz + 1.0]])
        did, _ = self.rings[0].pierced_by(B)
        self.assertFalse(did)

    def test_filter_returns_atom_globalidx_not_cell_idx(self):
        """Regression: nearby_atom_ids must return atom globalIdx values,
        not linkcell cell indices; confirm the values are in A['globalIdx']."""
        ci = int(self.A.loc[self.A['globalIdx'] == 7, 'linkcell_idx'].iloc[0])
        cell_set = self.lc.neighbor_cell_set(ci)
        atom_ids = self.lc.nearby_atom_ids(self.A, cell_set)
        valid = set(self.A['globalIdx'].values)
        self.assertTrue(set(atom_ids).issubset(valid),
                        f"nearby_atom_ids returned values not in globalIdx: "
                        f"{set(atom_ids) - valid}")


if __name__ == '__main__':
    unittest.main()
