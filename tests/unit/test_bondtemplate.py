"""

.. module:: test_bondtemplate
   :synopsis: unit tests for htpolynet.core.bondtemplate

.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import unittest
from copy import deepcopy

from htpolynet.core.bondtemplate import (
    BondTemplate, BondTemplateList,
    ReactionBond, ReactionBondList,
)


# ---------------------------------------------------------------------------
# Helpers — representative bond contexts
# ---------------------------------------------------------------------------

def _make_bt(names=None, resnames=None, intraresidue=False, order=1,
             bystander_resnames=None, bystander_atomnames=None,
             oneaway_resnames=None, oneaway_atomnames=None):
    """Return a BondTemplate with sensible defaults for any omitted field."""
    return BondTemplate(
        names              = names               or ['N1', 'C1'],
        resnames           = resnames            or ['GLY', 'ALA'],
        intraresidue       = intraresidue,
        order              = order,
        bystander_resnames = bystander_resnames  or [['SER'], []],
        bystander_atomnames= bystander_atomnames or [['CA'], []],
        oneaway_resnames   = oneaway_resnames    or [None, None],
        oneaway_atomnames  = oneaway_atomnames   or [None, None],
    )


def _make_rb(idx=None, resids=None, order=1,
             bystanders=None, bystanders_atomidx=None,
             oneaways=None, oneaways_atomidx=None):
    """Return a ReactionBond with sensible defaults for any omitted field."""
    return ReactionBond(
        idx               = idx                or [10, 20],
        resids            = resids             or [1, 2],
        order             = order,
        bystanders        = bystanders         or [[3], []],
        bystanders_atomidx= bystanders_atomidx or [[30], []],
        oneaways          = oneaways           or [None, None],
        oneaways_atomidx  = oneaways_atomidx   or [None, None],
    )


# ===========================================================================
# BondTemplate
# ===========================================================================

class TestBondTemplateConstruction(unittest.TestCase):
    """Attributes are set correctly by __init__."""

    def setUp(self):
        self.bt = _make_bt()

    def test_names_stored(self):
        self.assertEqual(self.bt.names, ['N1', 'C1'])

    def test_resnames_stored(self):
        self.assertEqual(self.bt.resnames, ['GLY', 'ALA'])

    def test_intraresidue_stored(self):
        self.assertFalse(self.bt.intraresidue)

    def test_order_stored(self):
        self.assertEqual(self.bt.order, 1)

    def test_bystander_resnames_stored(self):
        self.assertEqual(self.bt.bystander_resnames, [['SER'], []])

    def test_bystander_atomnames_stored(self):
        self.assertEqual(self.bt.bystander_atomnames, [['CA'], []])

    def test_oneaway_resnames_stored(self):
        self.assertEqual(self.bt.oneaway_resnames, [None, None])

    def test_oneaway_atomnames_stored(self):
        self.assertEqual(self.bt.oneaway_atomnames, [None, None])

    def test_intraresidue_true(self):
        bt = _make_bt(intraresidue=True)
        self.assertTrue(bt.intraresidue)

    def test_double_bond_order(self):
        bt = _make_bt(order=2)
        self.assertEqual(bt.order, 2)


class TestBondTemplateReverse(unittest.TestCase):
    """reverse() flips all parallel lists in place."""

    def setUp(self):
        self.bt = _make_bt(
            names               = ['N1', 'C1'],
            resnames            = ['GLY', 'ALA'],
            bystander_resnames  = [['SER'], ['THR']],
            bystander_atomnames = [['CA'], ['CB']],
            oneaway_resnames    = ['PRO', None],
            oneaway_atomnames   = ['CD', None],
        )
        self.original = deepcopy(self.bt)
        self.bt.reverse()

    def test_names_reversed(self):
        self.assertEqual(self.bt.names, ['C1', 'N1'])

    def test_resnames_reversed(self):
        self.assertEqual(self.bt.resnames, ['ALA', 'GLY'])

    def test_bystander_resnames_reversed(self):
        self.assertEqual(self.bt.bystander_resnames, [['THR'], ['SER']])

    def test_bystander_atomnames_reversed(self):
        self.assertEqual(self.bt.bystander_atomnames, [['CB'], ['CA']])

    def test_oneaway_resnames_reversed(self):
        self.assertEqual(self.bt.oneaway_resnames, [None, 'PRO'])

    def test_oneaway_atomnames_reversed(self):
        self.assertEqual(self.bt.oneaway_atomnames, [None, 'CD'])

    def test_order_unchanged_by_reverse(self):
        self.assertEqual(self.bt.order, self.original.order)

    def test_intraresidue_unchanged_by_reverse(self):
        self.assertEqual(self.bt.intraresidue, self.original.intraresidue)

    def test_double_reverse_is_identity(self):
        self.bt.reverse()  # reverse back
        self.assertEqual(self.bt, self.original)


class TestBondTemplateEquality(unittest.TestCase):
    """__eq__ compares all chemically meaningful fields."""

    def _bt(self, **kwargs):
        return _make_bt(**kwargs)

    def test_equal_to_itself(self):
        bt = self._bt()
        self.assertEqual(bt, bt)

    def test_equal_to_deep_copy(self):
        bt = self._bt()
        self.assertEqual(bt, deepcopy(bt))

    def test_different_names(self):
        self.assertNotEqual(self._bt(names=['N1', 'C1']), self._bt(names=['N2', 'C1']))

    def test_different_resnames(self):
        self.assertNotEqual(self._bt(resnames=['GLY', 'ALA']), self._bt(resnames=['GLY', 'SER']))

    def test_different_intraresidue(self):
        self.assertNotEqual(self._bt(intraresidue=False), self._bt(intraresidue=True))

    def test_different_bystander_resnames(self):
        self.assertNotEqual(
            self._bt(bystander_resnames=[['SER'], []]),
            self._bt(bystander_resnames=[['THR'], []]),
        )

    def test_different_bystander_atomnames(self):
        self.assertNotEqual(
            self._bt(bystander_atomnames=[['CA'], []]),
            self._bt(bystander_atomnames=[['CB'], []]),
        )

    def test_different_oneaway_resnames(self):
        self.assertNotEqual(
            self._bt(oneaway_resnames=[None, None]),
            self._bt(oneaway_resnames=['PRO', None]),
        )

    def test_order_not_in_eq(self):
        # __eq__ does not check order — verify the current behaviour
        bt1 = self._bt(order=1)
        bt2 = self._bt(order=2)
        self.assertEqual(bt1, bt2)


class TestBondTemplateIsReverseOf(unittest.TestCase):
    """is_reverse_of() detects reversed-direction equivalence."""

    def _asymmetric_bt(self, flip=False):
        bt = _make_bt(
            names               = ['N1', 'C1'],
            resnames            = ['GLY', 'ALA'],
            bystander_resnames  = [['SER'], ['THR']],
            bystander_atomnames = [['CA'], ['CB']],
            oneaway_resnames    = ['PRO', None],
            oneaway_atomnames   = ['CD', None],
        )
        if flip:
            bt.reverse()
        return bt

    def test_reversed_is_reverse_of_original(self):
        original = self._asymmetric_bt()
        reversed_bt = self._asymmetric_bt(flip=True)
        self.assertTrue(reversed_bt.is_reverse_of(original))

    def test_original_is_reverse_of_reversed(self):
        original = self._asymmetric_bt()
        reversed_bt = self._asymmetric_bt(flip=True)
        self.assertTrue(original.is_reverse_of(reversed_bt))

    def test_identical_is_not_reverse_of_itself_when_asymmetric(self):
        bt = self._asymmetric_bt()
        # bt is not its own reverse when the two ends differ
        self.assertFalse(bt.is_reverse_of(bt))

    def test_symmetric_bond_is_reverse_of_itself(self):
        # symmetric: both ends identical
        bt = _make_bt(
            names               = ['C1', 'C1'],
            resnames            = ['STY', 'STY'],
            bystander_resnames  = [[], []],
            bystander_atomnames = [[], []],
            oneaway_resnames    = [None, None],
            oneaway_atomnames   = [None, None],
        )
        self.assertTrue(bt.is_reverse_of(bt))

    def test_is_reverse_of_does_not_mutate_other(self):
        original = self._asymmetric_bt()
        snapshot = deepcopy(original)
        candidate = self._asymmetric_bt(flip=True)
        candidate.is_reverse_of(original)
        self.assertEqual(original, snapshot)

    def test_unrelated_bond_is_not_reverse(self):
        bt1 = _make_bt(names=['N1', 'C1'], resnames=['GLY', 'ALA'])
        bt2 = _make_bt(names=['O1', 'S1'], resnames=['CYS', 'MET'])
        self.assertFalse(bt1.is_reverse_of(bt2))


class TestBondTemplateStr(unittest.TestCase):
    """__str__ includes key identifiers."""

    def test_str_contains_names(self):
        bt = _make_bt(names=['N1', 'C1'])
        s = str(bt)
        self.assertIn('N1', s)
        self.assertIn('C1', s)

    def test_str_contains_resnames(self):
        bt = _make_bt(resnames=['GLY', 'ALA'])
        s = str(bt)
        self.assertIn('GLY', s)
        self.assertIn('ALA', s)


class TestBondTemplateList(unittest.TestCase):
    """BondTemplateList is a plain list of BondTemplate instances."""

    def test_is_list_subtype(self):
        btl: BondTemplateList = [_make_bt(), _make_bt(names=['O', 'N'])]
        self.assertIsInstance(btl, list)
        self.assertEqual(len(btl), 2)

    def test_membership(self):
        bt = _make_bt()
        btl: BondTemplateList = [bt]
        self.assertIn(bt, btl)


# ===========================================================================
# ReactionBond
# ===========================================================================

class TestReactionBondConstruction(unittest.TestCase):
    """Attributes are set correctly by __init__."""

    def setUp(self):
        self.rb = _make_rb()

    def test_idx_stored(self):
        self.assertEqual(self.rb.idx, [10, 20])

    def test_resids_stored(self):
        self.assertEqual(self.rb.resids, [1, 2])

    def test_order_stored(self):
        self.assertEqual(self.rb.order, 1)

    def test_bystander_resids_stored(self):
        self.assertEqual(self.rb.bystander_resids, [[3], []])

    def test_bystander_atomidx_stored(self):
        self.assertEqual(self.rb.bystander_atomidx, [[30], []])

    def test_oneaway_resids_stored(self):
        self.assertEqual(self.rb.oneaway_resids, [None, None])

    def test_oneaway_atomidx_stored(self):
        self.assertEqual(self.rb.oneaway_atomidx, [None, None])


class TestReactionBondReverse(unittest.TestCase):
    """reverse() flips all parallel lists in place."""

    def setUp(self):
        self.rb = _make_rb(
            idx                = [10, 20],
            resids             = [1, 2],
            bystanders         = [[3], [4, 5]],
            bystanders_atomidx = [[30], [40, 50]],
            oneaways           = [6, None],
            oneaways_atomidx   = [60, None],
        )
        self.original = deepcopy(self.rb)
        self.rb.reverse()

    def test_idx_reversed(self):
        self.assertEqual(self.rb.idx, [20, 10])

    def test_resids_reversed(self):
        self.assertEqual(self.rb.resids, [2, 1])

    def test_bystander_resids_reversed(self):
        self.assertEqual(self.rb.bystander_resids, [[4, 5], [3]])

    def test_bystander_atomidx_reversed(self):
        self.assertEqual(self.rb.bystander_atomidx, [[40, 50], [30]])

    def test_oneaway_resids_reversed(self):
        self.assertEqual(self.rb.oneaway_resids, [None, 6])

    def test_oneaway_atomidx_reversed(self):
        self.assertEqual(self.rb.oneaway_atomidx, [None, 60])

    def test_order_unchanged_by_reverse(self):
        self.assertEqual(self.rb.order, self.original.order)

    def test_double_reverse_restores_idx(self):
        self.rb.reverse()
        self.assertEqual(self.rb.idx, self.original.idx)

    def test_double_reverse_restores_resids(self):
        self.rb.reverse()
        self.assertEqual(self.rb.resids, self.original.resids)

    def test_double_reverse_restores_bystanders(self):
        self.rb.reverse()
        self.assertEqual(self.rb.bystander_resids, self.original.bystander_resids)

    def test_double_reverse_restores_oneaways(self):
        self.rb.reverse()
        self.assertEqual(self.rb.oneaway_resids, self.original.oneaway_resids)


class TestReactionBondStr(unittest.TestCase):
    """__str__ includes key identifiers."""

    def test_str_contains_idx(self):
        rb = _make_rb(idx=[10, 20])
        s = str(rb)
        self.assertIn('10', s)
        self.assertIn('20', s)

    def test_str_contains_resids(self):
        rb = _make_rb(resids=[1, 2])
        s = str(rb)
        self.assertIn('1', s)
        self.assertIn('2', s)


class TestReactionBondList(unittest.TestCase):
    """ReactionBondList is a plain list of ReactionBond instances."""

    def test_is_list_subtype(self):
        rbl: ReactionBondList = [_make_rb(), _make_rb(idx=[5, 6])]
        self.assertIsInstance(rbl, list)
        self.assertEqual(len(rbl), 2)


# ===========================================================================
# Relationship between BondTemplate and ReactionBond
# ===========================================================================

class TestBondTemplateReactionBondParallelism(unittest.TestCase):
    """BondTemplate and ReactionBond share the same parallel structure;
    reversing one should correspond to reversing the other."""

    def test_reverse_both_gives_consistent_direction(self):
        bt = _make_bt(names=['N1', 'C1'], resnames=['GLY', 'ALA'])
        rb = _make_rb(idx=[10, 20], resids=[1, 2])
        bt.reverse()
        rb.reverse()
        self.assertEqual(bt.names[0], 'C1')
        self.assertEqual(rb.idx[0], 20)

    def test_bt_is_reverse_of_detects_same_flip_as_rb_reverse(self):
        bt_fwd = _make_bt(
            names=['N1', 'C1'], resnames=['GLY', 'ALA'],
            bystander_resnames=[['SER'], []], bystander_atomnames=[['CA'], []],
        )
        bt_rev = deepcopy(bt_fwd)
        bt_rev.reverse()
        self.assertTrue(bt_rev.is_reverse_of(bt_fwd))
        self.assertTrue(bt_fwd.is_reverse_of(bt_rev))
