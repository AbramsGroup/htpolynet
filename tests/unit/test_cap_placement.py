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
    _report_placements, _placement_neighbors, _angle_window,
    _MIN_COC_ANGLE, _MAX_COC_ANGLE,
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
    """The atoms a cap is bonded through must not be scored against.

    This is the configuration the rest of this module's fixtures lack, and its
    absence let a released version report a clearance that was pinned at the
    O-C bond length on every cap in every run.  The attachment oxygen is the
    hard case -- it fixes the metric at exactly `oc_len` -- and its aryl carbon
    is the soft one, bounding the metric at the C-O-C geometry so that a
    comfortably placed cap reports a constant instead of a measurement.
    """
    def setUp(self):
        self.o_idx = 7
        self.aryl_idx = 8
        self.o = np.array([5.0, 5.0, 5.0])
        self.h = self.o + np.array([0.1, 0.0, 0.0])       # O-H along +x
        self.aryl = self.o - np.array([OC_LEN, 0.0, 0.0])  # aryl C along -x
        # a real neighbourhood: the bonded O, its aryl carbon, and one
        # unrelated atom sitting in the O-H direction
        self.background = np.array([self.o, self.aryl, self.o + np.array([0.20, 0.0, 0.0])])
        self.background_idx = np.array([self.o_idx, self.aryl_idx, 900])
        self.bonded = (self.o_idx, self.aryl_idx)

    def _local(self, idx=None, bonded=None, bg=None):
        return _placement_neighbors(self.o,
                                    self.background if bg is None else bg,
                                    self.background_idx if idx is None else idx,
                                    BOX,
                                    bonded=self.bonded if bonded is None else bonded)

    def test_drops_the_bonded_oxygen_and_its_aryl_carbon(self):
        local = self._local()
        self.assertEqual(len(local), 1)
        self.assertFalse(np.any(np.all(np.isclose(local, self.o), axis=1)))
        self.assertFalse(np.any(np.all(np.isclose(local, self.aryl), axis=1)))

    def test_clearance_is_no_longer_pinned_at_the_bond_length(self):
        # with the bonded O left in, every candidate direction scores exactly
        # oc_len, the target is unreachable by construction, and the metric
        # carries no information about crowding
        pinned = _choose_cap_placement(self.o, self.h, self.background, BOX)
        self.assertAlmostEqual(pinned['clearance'], OC_LEN, places=6)
        self.assertLess(pinned['clearance'], 0.15)
        # with it dropped, the search can actually discriminate
        real = _choose_cap_placement(self.o, self.h, self._local(), BOX,
                                     anchor_xyz=self.aryl)
        self.assertGreater(real['clearance'], OC_LEN)
        self.assertGreaterEqual(real['clearance'], 0.15)

    def test_the_aryl_carbon_would_saturate_the_metric(self):
        # left in, it bounds clearance at the C-O-C geometry (0.272 nm at
        # 180 deg), so any cap with no real neighbour nearby reports that
        # constant rather than a measurement and the median stops varying
        with_aryl = _placement_neighbors(self.o, np.array([self.o, self.aryl]),
                                         np.array([self.o_idx, self.aryl_idx]),
                                         BOX, bonded=(self.o_idx,))
        capped = _choose_cap_placement(self.o, self.h, with_aryl, BOX, target=0.40)
        self.assertLessEqual(capped['clearance'], 2 * OC_LEN + 1e-6)
        # dropped, an empty neighbourhood reads as empty
        without = _placement_neighbors(self.o, np.array([self.o, self.aryl]),
                                       np.array([self.o_idx, self.aryl_idx]),
                                       BOX, bonded=self.bonded)
        self.assertEqual(len(without), 0)

    def test_keeps_another_caps_oxygen(self):
        # only this cap's own bonded atoms go; the others are real obstacles
        local = self._local(idx=np.array([99, 100, 900]))
        self.assertEqual(len(local), 3)

    def test_keeps_already_placed_caps(self):
        bg = np.vstack([self.background, self.o + np.array([0.0, 0.3, 0.0])])
        idx = np.concatenate([self.background_idx, [-1]])
        local = self._local(idx=idx, bg=bg)
        self.assertEqual(len(local), 2)

    def test_still_drops_the_distant(self):
        bg = np.vstack([self.background, self.o + np.array([2.0, 0.0, 0.0])])
        idx = np.concatenate([self.background_idx, [1000]])
        self.assertEqual(len(self._local(idx=idx, bg=bg)), 1)

    def test_tolerates_a_missing_anchor(self):
        # an O with no heavy partner shouldn't crash the exclusion
        local = self._local(bonded=(self.o_idx, None))
        self.assertEqual(len(local), 2)

    def test_empty_background(self):
        out = _placement_neighbors(self.o, np.empty((0, 3)), np.empty(0, dtype=int),
                                   BOX, bonded=self.bonded)
        self.assertEqual(len(out), 0)


class TestAngleWindow(unittest.TestCase):
    """Clearance cannot express a bond angle, so the angle is stated separately."""
    def setUp(self):
        self.o = np.array([5.0, 5.0, 5.0])
        self.aryl = self.o - np.array([OC_LEN, 0.0, 0.0])   # anchor along -x

    def _mask(self, angles_deg):
        # direction at angle theta from the O->aryl vector (which points -x)
        d = np.array([[-np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0]
                      for a in angles_deg])
        return _angle_window(d, self.o, self.aryl)

    def test_rejects_fold_back_onto_the_ring(self):
        self.assertFalse(self._mask([0.0, 30.0, 60.0, 89.0]).any())

    def test_rejects_the_linear_ether(self):
        self.assertFalse(self._mask([151.0, 170.0, 180.0]).any())

    def test_accepts_the_chemical_angle(self):
        self.assertTrue(self._mask([109.0, 118.0, 120.0, 130.0]).all())

    def test_bounds_are_inclusive(self):
        self.assertTrue(self._mask([_MIN_COC_ANGLE + 1e-6, _MAX_COC_ANGLE - 1e-6]).all())

    def test_no_anchor_allows_everything(self):
        d = _sphere_directions(48)
        self.assertTrue(_angle_window(d, self.o, None).all())

    def test_leaves_a_usable_fraction_of_the_sphere(self):
        # the search needs somewhere to go; the window is a spherical zone of
        # solid-angle fraction (cos 90 - cos 150)/2 = 0.43
        d = _sphere_directions(400)
        frac = _angle_window(d, self.o, self.aryl).mean()
        self.assertAlmostEqual(frac, 0.433, delta=0.03)


class TestAngularGuardInPlacement(unittest.TestCase):
    def setUp(self):
        self.o = np.array([5.0, 5.0, 5.0])
        self.aryl = self.o - np.array([OC_LEN, 0.0, 0.0])

    def _coc_angle(self, c_xyz):
        u = (np.asarray(c_xyz) - self.o) / np.linalg.norm(np.asarray(c_xyz) - self.o)
        a = (self.aryl - self.o) / np.linalg.norm(self.aryl - self.o)
        return float(np.degrees(np.arccos(np.clip(u @ a, -1.0, 1.0))))

    def test_open_space_would_go_linear_without_the_guard(self):
        # nothing in the box: the first direction that reaches an unreachable
        # target wins, and unguarded that can be any angle at all
        h = self.o + np.array([0.0, 0.1, 0.0])   # O-H at 90 deg, just outside
        free = _choose_cap_placement(self.o, h, np.empty((0, 3)), BOX,
                                     anchor_xyz=self.aryl)
        ang = self._coc_angle(free['c'])
        self.assertGreaterEqual(ang, _MIN_COC_ANGLE)
        self.assertLessEqual(ang, _MAX_COC_ANGLE)

    def test_a_blocked_cap_still_lands_in_the_window(self):
        h = self.o + np.array([0.1, 0.0, 0.0])
        blockers = self.o + np.array([[0.20, 0.0, 0.0], [0.0, 0.20, 0.0]])
        placed = _choose_cap_placement(self.o, h, blockers, BOX, anchor_xyz=self.aryl)
        self.assertFalse(placed['kept_o_h_direction'])
        ang = self._coc_angle(placed['c'])
        self.assertGreaterEqual(ang, _MIN_COC_ANGLE)
        self.assertLessEqual(ang, _MAX_COC_ANGLE)

    def test_an_out_of_window_preferred_direction_is_not_kept(self):
        # the fallback direction used when an O has no H can point anywhere;
        # clear space alone must not be enough to keep it
        bad = self.o + np.array([-0.1, 0.0, 0.0])     # straight at the aryl C
        placed = _choose_cap_placement(self.o, bad, np.empty((0, 3)), BOX,
                                       anchor_xyz=self.aryl)
        self.assertFalse(placed['kept_o_h_direction'])
        self.assertGreaterEqual(self._coc_angle(placed['c']), _MIN_COC_ANGLE)

    def test_no_anchor_reproduces_the_unguarded_search(self):
        h = self.o + np.array([0.1, 0.0, 0.0])
        blockers = self.o + np.array([[0.20, 0.0, 0.0]])
        a = _choose_cap_placement(self.o, h, blockers, BOX)
        b = _choose_cap_placement(self.o, h, blockers, BOX, anchor_xyz=None)
        np.testing.assert_allclose(a['c'], b['c'])

    def test_bond_lengths_survive_the_guard(self):
        h = self.o + np.array([0.0, 0.1, 0.0])
        p = _choose_cap_placement(self.o, h, np.empty((0, 3)), BOX, anchor_xyz=self.aryl)
        self.assertAlmostEqual(np.linalg.norm(p['c'] - self.o), OC_LEN, places=9)
        self.assertAlmostEqual(np.linalg.norm(p['n'] - p['c']), 0.116, places=9)


class TestOutOfAngleReporting(unittest.TestCase):
    """The two reasons a preferred direction is abandoned must stay separable.

    2.6.0 shipped a metric that rejected every preferred direction for a
    reason unrelated to crowding, and the output read as a crowded box.  If
    the angle window ever bites the O-H vector systematically it produces the
    same symptom, so it is counted separately.
    """
    def _p(self, in_window, clearance=0.2):
        return {'o': 1, 'clearance': clearance, 'kept_o_h_direction': False,
                'blind_clearance': 0.05, 'blind_in_window': in_window}

    def test_counts_the_out_of_window_ones(self):
        out = _report_placements([self._p(True), self._p(False), self._p(False)], 0.15)
        self.assertEqual(out['n_preferred_out_of_angle'], 2)
        self.assertEqual(out['n_direction_searched'], 3)

    def test_zero_when_every_preferred_direction_was_placeable(self):
        out = _report_placements([self._p(True), self._p(True)], 0.15)
        self.assertEqual(out['n_preferred_out_of_angle'], 0)

    def test_defaults_to_in_window_for_records_without_the_key(self):
        rec = self._p(True)
        del rec['blind_in_window']
        self.assertEqual(_report_placements([rec], 0.15)['n_preferred_out_of_angle'], 0)

    def test_empty(self):
        self.assertEqual(_report_placements([], 0.15)['n_preferred_out_of_angle'], 0)


if __name__ == '__main__':
    unittest.main()
