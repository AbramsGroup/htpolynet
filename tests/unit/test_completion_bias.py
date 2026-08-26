"""

.. module:: test_completion_bias
   :synopsis: tests the CURE.controls.completion_bias candidate ranking

.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import unittest
import logging
logger=logging.getLogger(__name__)
import pandas as pd
from htpolynet.cure.curecontroller import CureController, rank_bond_candidates, residue_reaction_counts, residue_functionality

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

def atoms(rows):
    """Builds an atom frame from (resNum, z, nreactions) triples."""
    return pd.DataFrame({
        'resNum':    [r[0] for r in rows],
        'z':         [r[1] for r in rows],
        'nreactions':[r[2] for r in rows],
    })

class TestResidueFunctionality(unittest.TestCase):
    def test_sum_is_conserved_as_the_cure_proceeds(self):
        # a triazine starts with three z=1 ring carbons; two have since
        # reacted.  Its functionality must still read 3.
        fresh=atoms([(1,1,0),(1,1,0),(1,1,0)])
        partly=atoms([(1,0,1),(1,0,1),(1,1,0)])
        self.assertEqual(residue_functionality(fresh)[1],3)
        self.assertEqual(residue_functionality(partly)[1],3)

    def test_bridge_and_crosslinker_are_distinguished(self):
        adf=atoms([(1,1,0),(1,1,0),          # a difunctional bridge
                   (2,1,0),(2,1,0),(2,1,0)]) # a trifunctional crosslinker
        func=residue_functionality(adf)
        self.assertEqual(func[1],2)
        self.assertEqual(func[2],3)

class TestBiasSideCheck(unittest.TestCase):
    """The failure mode is a crosslinker declared as reactant A: the bias then
    completes bridges instead.  It cannot be caught by looking for an all-zero
    bias key, because a difunctional bridge accumulates reactions too."""

    def setUp(self):
        self.cc=CureController({'controls':{'completion_bias':True}})
        # residue 1 is a difunctional bridge, residue 2 a trifunctional crosslinker
        self.adf=atoms([(1,1,0),(1,1,0), (2,1,0),(2,1,0),(2,1,0)])

    def test_silent_when_the_crosslinker_is_on_the_b_side(self):
        bdf=pd.DataFrame({'ri':[1],'rj':[2],'r':[0.3]})
        with self.assertNoLogs('htpolynet.cure.curecontroller',level='WARNING'):
            self.cc._check_bias_side(self.adf,bdf)

    def test_warns_when_the_crosslinker_is_on_the_a_side(self):
        bdf=pd.DataFrame({'ri':[2],'rj':[1],'r':[0.3]})
        with self.assertLogs('htpolynet.cure.curecontroller',level='WARNING') as cm:
            self.cc._check_bias_side(self.adf,bdf)
        self.assertIn('3 reactive sites vs 2',''.join(cm.output))

    def test_warns_only_once_per_run(self):
        bdf=pd.DataFrame({'ri':[2],'rj':[1],'r':[0.3]})
        with self.assertLogs('htpolynet.cure.curecontroller',level='WARNING'):
            self.cc._check_bias_side(self.adf,bdf)
        with self.assertNoLogs('htpolynet.cure.curecontroller',level='WARNING'):
            self.cc._check_bias_side(self.adf,bdf)

    def test_symmetric_functionality_is_not_a_complaint(self):
        adf=atoms([(1,1,0),(1,1,0), (2,1,0),(2,1,0)])
        bdf=pd.DataFrame({'ri':[2],'rj':[1],'r':[0.3]})
        with self.assertNoLogs('htpolynet.cure.curecontroller',level='WARNING'):
            self.cc._check_bias_side(adf,bdf)

    def test_no_crash_when_the_grx_predates_these_attributes(self):
        adf=pd.DataFrame({'resNum':[1,2]})
        bdf=pd.DataFrame({'ri':[2],'rj':[1],'r':[0.3]})
        self.cc._check_bias_side(adf,bdf)
        self.assertFalse(self.cc.bias_side_checked)
