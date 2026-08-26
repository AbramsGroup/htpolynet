"""

.. module:: test_completion_bias
   :synopsis: tests the CURE.controls.completion_bias candidate ranking

.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import unittest
import logging
logger=logging.getLogger(__name__)
import pandas as pd
from htpolynet.cure.curecontroller import CureController, rank_bond_candidates, residue_reaction_counts

def candidates(rows):
    """Builds a minimal bond-candidate frame: (rj, r) pairs with the columns
    the ranking actually reads, plus one it must carry through untouched."""
    return pd.DataFrame({
        'ai':   [10*i+1 for i in range(len(rows))],
        'aj':   [10*i+2 for i in range(len(rows))],
        'ri':   [100+i  for i in range(len(rows))],
        'rj':   [r[0]   for r in rows],
        'r':    [r[1]   for r in rows],
        'order':[1      for _ in rows],
    })

class TestRankBondCandidates(unittest.TestCase):
    def test_no_counts_is_distance_only(self):
        bdf=candidates([(1,0.44),(2,0.31),(3,0.52),(4,0.29)])
        ranked=rank_bond_candidates(bdf)
        self.assertEqual(ranked['r'].to_list(),[0.29,0.31,0.44,0.52])
        self.assertEqual(ranked['rj'].to_list(),[4,2,1,3])

    def test_bias_promotes_the_most_complete_crosslinker(self):
        # residue 3 already carries 2 of its 3 bonds but is the farthest
        # candidate; residue 4 is unreacted and the nearest.  Distance alone
        # picks 4 -- which is exactly the mechanism that spreads bonds over
        # every crosslinker instead of completing any of them.
        bdf=candidates([(1,0.44),(2,0.31),(3,0.52),(4,0.29)])
        counts=pd.Series({1:1,2:0,3:2,4:0})
        ranked=rank_bond_candidates(bdf,counts)
        self.assertEqual(ranked['rj'].to_list(),[3,1,4,2])

    def test_bias_breaks_ties_by_distance_within_a_group(self):
        bdf=candidates([(1,0.44),(2,0.31),(3,0.52),(4,0.29)])
        counts=pd.Series({1:2,2:2,3:1,4:1})
        ranked=rank_bond_candidates(bdf,counts)
        # both 2-bond residues first, nearer one leading; then both 1-bond
        self.assertEqual(ranked['rj'].to_list(),[2,1,4,3])

    def test_residue_absent_from_counts_ranks_as_unreacted(self):
        bdf=candidates([(1,0.44),(9,0.10)])
        counts=pd.Series({1:2})
        ranked=rank_bond_candidates(bdf,counts)
        self.assertEqual(ranked['rj'].to_list(),[1,9])

    def test_ranking_leaves_the_schema_alone(self):
        bdf=candidates([(1,0.44),(2,0.31)])
        counts=pd.Series({1:2,2:0})
        ranked=rank_bond_candidates(bdf,counts)
        self.assertEqual(list(ranked.columns),list(bdf.columns))
        self.assertEqual(list(ranked.index),[0,1])

class TestResidueReactionCounts(unittest.TestCase):
    def test_counts_sum_over_a_residues_atoms(self):
        # a triazine's three ring carbons each carry at most one bond, so the
        # residue sum is the number of bridges attached to it
        adf=pd.DataFrame({
            'resNum':    [1,1,1,1,1,1, 2,2,2,2,2,2],
            'nreactions':[1,1,0,0,0,0, 1,0,0,0,0,0],
        })
        counts=residue_reaction_counts(adf)
        self.assertEqual(counts[1],2)
        self.assertEqual(counts[2],1)

class TestCompletionBiasControl(unittest.TestCase):
    def test_default_is_off(self):
        cc=CureController({})
        self.assertFalse(cc.dicts['controls']['completion_bias'])
        adf=pd.DataFrame({'resNum':[1],'nreactions':[0]})
        self.assertIsNone(cc._completion_bias_counts(adf))

    def test_enabled_yields_counts(self):
        cc=CureController({'controls':{'completion_bias':True}})
        adf=pd.DataFrame({'resNum':[1,1,2],'nreactions':[1,1,0]})
        counts=cc._completion_bias_counts(adf)
        self.assertIsNotNone(counts)
        self.assertEqual(counts[1],2)

    def test_missing_nreactions_falls_back_to_distance(self):
        # a .grx written before this attribute existed must not crash a cure
        cc=CureController({'controls':{'completion_bias':True}})
        adf=pd.DataFrame({'resNum':[1,2]})
        with self.assertLogs('htpolynet.cure.curecontroller',level='WARNING'):
            self.assertIsNone(cc._completion_bias_counts(adf))
