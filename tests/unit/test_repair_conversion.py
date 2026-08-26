"""

.. module:: test_repair_conversion
   :synopsis: tests the crosslinker-conversion figure reported after postcure repair

.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import unittest
import logging
logger=logging.getLogger(__name__)
from htpolynet.repair.cyanate_cap import _completion_stats

class TestCompletionStats(unittest.TestCase):
    def test_partial_cure(self):
        # a partly-cured BADCy box: 240 triazines, 58 dismantled by repair
        st=_completion_stats('TAZ',240,58)
        self.assertEqual(st['n_complete'],182)
        self.assertEqual(st['n_dismantled'],58)
        self.assertAlmostEqual(st['crosslinker_conversion'],182/240)

    def test_full_cure(self):
        st=_completion_stats('TAZ',240,0)
        self.assertEqual(st['crosslinker_conversion'],1.0)

    def test_no_crosslinkers_does_not_divide_by_zero(self):
        st=_completion_stats('TAZ',0,0)
        self.assertEqual(st['crosslinker_conversion'],0.0)

    def test_reports_the_residue_it_was_given(self):
        st=_completion_stats('XYZ',10,3)
        self.assertEqual(st['residue'],'XYZ')


class TestRunRepairWiring(unittest.TestCase):
    """The (total, stats) contract between run_repair and the runtime is only
    otherwise exercised by a full build, so pin it here."""

    def test_no_specs(self):
        from htpolynet.repair import run_repair
        total, stats = run_repair(None, None, [], None)
        self.assertEqual(total, 0)
        self.assertEqual(stats, [])

    def test_unknown_type_is_skipped_not_fatal(self):
        from htpolynet.repair import run_repair
        with self.assertLogs('htpolynet.repair', level='WARNING'):
            total, stats = run_repair(None, None, [{'type': 'no_such_driver'}], None)
        self.assertEqual(total, 0)
        self.assertEqual(stats, [])

    def test_driver_result_becomes_total_and_stats(self):
        import htpolynet.repair.cyanate_cap as cc
        from htpolynet.repair import run_repair
        recorded = _completion_stats('TAZ', 240, 58, log=False)
        original = cc.triazine_to_cyanate_cap
        cc.triazine_to_cyanate_cap = lambda *a, **k: recorded
        try:
            total, stats = run_repair(None, None, [{'type': 'triazine_to_cyanate_cap'}], None)
        finally:
            cc.triazine_to_cyanate_cap = original
        self.assertEqual(total, 58)
        self.assertEqual(stats, [recorded])
