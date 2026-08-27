"""

.. module:: test_cap_placement
   :synopsis: tests neighbourhood-aware placement of transferred -C#N caps

.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import unittest
import logging
logger=logging.getLogger(__name__)
import numpy as np
from htpolynet.repair.cyanate_cap import (
    _sphere_directions, _clearance, _choose_cap_placement, _place_cyn_along,
    _report_placements, _placement_neighbors,
)

OC_LEN = 0.136

BOX = np.array([10.0, 10.0, 10.0])

class TestSphereDirections(unittest.TestCase):
    def test_unit_vectors(self):
        d = _sphere_directions(48)
        self.assertEqual(d.shape, (48, 3))
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)

    def test_deterministic(self):
        # a rebuild of the same system must place its caps identically
        np.testing.assert_array_equal(_sphere_directions(48), _sphere_directions(48))

    def test_directions_span_the_sphere(self):
        # every axis, both signs, is reachable -- otherwise a blocked cap has
        # nowhere to go in whole regions of space
        d = _sphere_directions(48)
        for axis in range(3):
            self.assertGreater(d[:, axis].max(), 0.8)
            self.assertLess(d[:, axis].min(), -0.8)

class TestClearance(unittest.TestCase):
    def test_no_neighbors_is_infinite(self):
        self.assertEqual(_clearance(np.zeros(3), np.ones(3), np.empty((0, 3)), BOX), float('inf'))

    def test_takes_the_closest_of_either_atom(self):
        c = np.array([0.0, 0.0, 0.0])
        n = np.array([0.5, 0.0, 0.0])
        near_n = np.array([[0.6, 0.0, 0.0]])
        self.assertAlmostEqual(_clearance(c, n, near_n, BOX), 0.1)

    def test_uses_the_minimum_image(self):
        c = np.array([0.05, 0.0, 0.0])
        n = np.array([0.05, 0.5, 0.0])
        across = np.array([[9.95, 0.0, 0.0]])   # 0.1 nm away through the wall
        self.assertAlmostEqual(_clearance(c, n, across, BOX), 0.1)

class TestChooseCapPlacement(unittest.TestCase):
    def setUp(self):
        self.o = np.array([5.0, 5.0, 5.0])
        self.h = self.o + np.array([0.1, 0.0, 0.0])   # O-H points along +x

    def test_open_space_keeps_the_old_behavior(self):
        r = _choose_cap_placement(self.o, self.h, np.empty((0, 3)), BOX)
        self.assertTrue(r['kept_o_h_direction'])
        c0, n0 = _place_cyn_along(self.o, self.h)
        np.testing.assert_allclose(r['c'], c0)
        np.testing.assert_allclose(r['n'], n0)

    def test_bond_lengths_are_held_fixed(self):
        blocker = np.array([[5.2, 5.0, 5.0]])
        r = _choose_cap_placement(self.o, self.h, blocker, BOX)
        self.assertFalse(r['kept_o_h_direction'])
        self.assertAlmostEqual(float(np.linalg.norm(r['c'] - self.o)), 0.136, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(r['n'] - r['c'])), 0.116, places=9)

    def test_moves_away_from_an_occupied_direction(self):
        # an atom sitting exactly where the cap would go along the O-H vector
        blocker = np.array([[5.2, 5.0, 5.0]])
        c_blind, n_blind = _place_cyn_along(self.o, self.h)
        blind_gap = _clearance(c_blind, n_blind, blocker, BOX)
        r = _choose_cap_placement(self.o, self.h, blocker, BOX)
        self.assertGreater(r['clearance'], blind_gap)
        self.assertGreaterEqual(r['clearance'], 0.15)

    def test_reports_what_the_blind_direction_would_have_given(self):
        # free to compute, and the only way to measure on a real box what
        # placement was doing before the direction search existed
        blocker = np.array([[5.2, 5.0, 5.0]])
        c_blind, n_blind = _place_cyn_along(self.o, self.h)
        r = _choose_cap_placement(self.o, self.h, blocker, BOX)
        self.assertAlmostEqual(r['blind_clearance'],
                               _clearance(c_blind, n_blind, blocker, BOX))
        self.assertLess(r['blind_clearance'], r['clearance'])

    def test_reports_the_best_it_could_do_when_boxed_in(self):
        # a shell of atoms all around: no direction is clear, and the caller
        # needs to hear that rather than get a confident bad placement
        shell = self.o + _sphere_directions(60) * 0.2
        r = _choose_cap_placement(self.o, self.h, shell, BOX)
        self.assertFalse(r['kept_o_h_direction'])
        self.assertLess(r['clearance'], 0.15)
        self.assertGreater(r['clearance'], 0.0)

class TestReportPlacements(unittest.TestCase):
    def test_empty(self):
        summary = _report_placements([], 0.22)
        self.assertEqual(summary['n_transferred'], 0)
        self.assertIsNone(summary['min_clearance_nm'])

    def test_counts_and_stays_quiet_when_all_are_clear(self):
        placements = [{'o': i, 'clearance': 0.30, 'kept_o_h_direction': True,
                       'blind_clearance': 0.30} for i in range(3)]
        with self.assertNoLogs('htpolynet.repair.cyanate_cap', level='WARNING'):
            summary = _report_placements(placements, 0.22)
        self.assertEqual(summary['n_transferred'], 3)
        self.assertEqual(summary['n_below_target'], 0)
        self.assertEqual(summary['n_direction_searched'], 0)
        self.assertAlmostEqual(summary['min_clearance_nm'], 0.30)
        self.assertAlmostEqual(summary['median_clearance_nm'], 0.30)
        self.assertAlmostEqual(summary['blind_median_clearance_nm'], 0.30)

    def test_names_the_tight_ones(self):
        placements = [{'o': 7, 'clearance': 0.05, 'kept_o_h_direction': False,
                       'blind_clearance': 0.01},
                      {'o': 8, 'clearance': 0.30, 'kept_o_h_direction': True,
                       'blind_clearance': 0.30}]
        with self.assertLogs('htpolynet.repair.cyanate_cap', level='WARNING') as cm:
            summary = _report_placements(placements, 0.22)
        text = ''.join(cm.output)
        self.assertIn('O 7', text)
        self.assertIn('Lennard-Jones', text)
        self.assertEqual(summary['n_below_target'], 1)
        self.assertEqual(summary['n_direction_searched'], 1)
        self.assertEqual(summary['n_blind_would_overlap'], 1)
        self.assertAlmostEqual(summary['blind_min_clearance_nm'], 0.01)
        # a median, not just the extreme, is what survives averaging over runs
        self.assertAlmostEqual(summary['median_clearance_nm'], 0.175)
        self.assertAlmostEqual(summary['blind_median_clearance_nm'], 0.155)

class TestPlacementNeighbors(unittest.TestCase):
    """The attachment oxygen is bonded to the cap, and must not be scored against.

    This is the configuration the rest of this module's fixtures lack, and its
    absence let a released version report a clearance that was pinned at the
    O-C bond length on every cap in every run.  The first test below fails
    against that version.
    """
    def setUp(self):
        self.o_idx = 7
        self.o = np.array([5.0, 5.0, 5.0])
        self.h = self.o + np.array([0.1, 0.0, 0.0])       # O-H along +x
        self.aryl = self.o - np.array([OC_LEN, 0.0, 0.0])  # aryl C along -x
        # a real neighbourhood: the bonded O, its aryl carbon, and one
        # unrelated atom sitting in the O-H direction
        self.background = np.array([self.o, self.aryl, self.o + np.array([0.20, 0.0, 0.0])])
        self.background_idx = np.array([self.o_idx, 8, 900])

    def test_drops_the_bonded_oxygen(self):
        local = _placement_neighbors(self.o_idx, self.o, self.background,
                                     self.background_idx, BOX)
        self.assertEqual(len(local), 2)
        self.assertFalse(np.any(np.all(np.isclose(local, self.o), axis=1)))

    def test_clearance_is_no_longer_pinned_at_the_bond_length(self):
        # with the bonded O left in, every candidate direction scores exactly
        # oc_len, the target is unreachable by construction, and the metric
        # carries no information about crowding
        pinned = _choose_cap_placement(self.o, self.h, self.background, BOX)
        self.assertAlmostEqual(pinned['clearance'], OC_LEN, places=6)
        self.assertLess(pinned['clearance'], 0.15)
        # with it dropped, the search can actually discriminate
        local = _placement_neighbors(self.o_idx, self.o, self.background,
                                     self.background_idx, BOX)
        real = _choose_cap_placement(self.o, self.h, local, BOX)
        self.assertGreater(real['clearance'], OC_LEN)
        self.assertGreaterEqual(real['clearance'], 0.15)

    def test_keeps_the_aryl_carbon(self):
        # nothing else stops the search folding the cap back onto the ring
        local = _placement_neighbors(self.o_idx, self.o, self.background,
                                     self.background_idx, BOX)
        self.assertTrue(np.any(np.all(np.isclose(local, self.aryl), axis=1)))

    def test_the_aryl_carbon_caps_what_is_reachable(self):
        # in vacuum but for the aryl carbon, the best any direction can do is
        # 0.272 nm -- a target at or above that would flag every cap
        local = _placement_neighbors(self.o_idx, self.o,
                                     np.array([self.o, self.aryl]),
                                     np.array([self.o_idx, 8]), BOX)
        blocked = _choose_cap_placement(self.o, self.o + np.array([0.001, 0, 0]),
                                        local, BOX, target=0.40)
        self.assertLessEqual(blocked['clearance'], 2 * OC_LEN + 1e-6)

    def test_keeps_another_caps_oxygen(self):
        # only this cap's own O is bonded to it; the others are real obstacles
        local = _placement_neighbors(self.o_idx, self.o, self.background,
                                     np.array([99, 8, 900]), BOX)
        self.assertEqual(len(local), 3)

    def test_keeps_already_placed_caps(self):
        # placed cap atoms carry -1 and must never be masked out
        bg = np.vstack([self.background, self.o + np.array([0.0, 0.3, 0.0])])
        idx = np.concatenate([self.background_idx, [-1]])
        local = _placement_neighbors(self.o_idx, self.o, bg, idx, BOX)
        self.assertEqual(len(local), 3)

    def test_still_drops_the_distant(self):
        far = np.vstack([self.background, self.o + np.array([2.0, 0.0, 0.0])])
        idx = np.concatenate([self.background_idx, [1000]])
        local = _placement_neighbors(self.o_idx, self.o, far, idx, BOX)
        self.assertEqual(len(local), 2)

    def test_empty_background(self):
        out = _placement_neighbors(self.o_idx, self.o, np.empty((0, 3)),
                                   np.empty(0, dtype=int), BOX)
        self.assertEqual(len(out), 0)

if __name__ == '__main__':
    unittest.main()
