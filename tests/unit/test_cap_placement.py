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
    _report_placements,
)

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
        c, n, gap, kept = _choose_cap_placement(self.o, self.h, np.empty((0, 3)), BOX)
        self.assertTrue(kept)
        c0, n0 = _place_cyn_along(self.o, self.h)
        np.testing.assert_allclose(c, c0)
        np.testing.assert_allclose(n, n0)

    def test_bond_lengths_are_held_fixed(self):
        blocker = np.array([[5.2, 5.0, 5.0]])
        c, n, gap, kept = _choose_cap_placement(self.o, self.h, blocker, BOX)
        self.assertFalse(kept)
        self.assertAlmostEqual(float(np.linalg.norm(c - self.o)), 0.136, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(n - c)), 0.116, places=9)

    def test_moves_away_from_an_occupied_direction(self):
        # an atom sitting exactly where the cap would go along the O-H vector
        blocker = np.array([[5.2, 5.0, 5.0]])
        c_blind, n_blind = _place_cyn_along(self.o, self.h)
        blind_gap = _clearance(c_blind, n_blind, blocker, BOX)
        c, n, gap, kept = _choose_cap_placement(self.o, self.h, blocker, BOX)
        self.assertGreater(gap, blind_gap)
        self.assertGreaterEqual(gap, 0.22)

    def test_reports_the_best_it_could_do_when_boxed_in(self):
        # a shell of atoms all around: no direction is clear, and the caller
        # needs to hear that rather than get a confident bad placement
        shell = self.o + _sphere_directions(60) * 0.2
        c, n, gap, kept = _choose_cap_placement(self.o, self.h, shell, BOX)
        self.assertFalse(kept)
        self.assertLess(gap, 0.22)
        self.assertGreater(gap, 0.0)

class TestReportPlacements(unittest.TestCase):
    def test_empty(self):
        summary = _report_placements([], 0.22)
        self.assertEqual(summary['n_transferred'], 0)
        self.assertIsNone(summary['min_clearance_nm'])

    def test_counts_and_stays_quiet_when_all_are_clear(self):
        placements = [{'o': i, 'clearance': 0.30, 'kept_o_h_direction': True} for i in range(3)]
        with self.assertNoLogs('htpolynet.repair.cyanate_cap', level='WARNING'):
            summary = _report_placements(placements, 0.22)
        self.assertEqual(summary['n_transferred'], 3)
        self.assertEqual(summary['n_below_target'], 0)
        self.assertEqual(summary['n_direction_searched'], 0)
        self.assertAlmostEqual(summary['min_clearance_nm'], 0.30)

    def test_names_the_tight_ones(self):
        placements = [{'o': 7, 'clearance': 0.05, 'kept_o_h_direction': False},
                      {'o': 8, 'clearance': 0.30, 'kept_o_h_direction': True}]
        with self.assertLogs('htpolynet.repair.cyanate_cap', level='WARNING') as cm:
            summary = _report_placements(placements, 0.22)
        text = ''.join(cm.output)
        self.assertIn('O 7', text)
        self.assertIn('Lennard-Jones', text)
        self.assertEqual(summary['n_below_target'], 1)
        self.assertEqual(summary['n_direction_searched'], 1)
