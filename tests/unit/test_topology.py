"""

.. module:: test_topology
   :synopsis: tests topology
   
.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import unittest
import logging
logger=logging.getLogger(__name__)
import htpolynet.core.topology as tp
import os
import tempfile
import pandas as pd
import numpy as np

class TestTopology(unittest.TestCase):
    def test_read_top(self):
        fn='test.top' # this top file has blank entries for all bond, angle, and dihedral parameters
        T=tp.Topology.read_top(fn) # blanks are padded
        self.assertTrue('bonds' in T.D)
        self.assertEqual(tp._PAD_,-99.99)
        self.assertTrue(all(T.D['bonds']['c1']==tp._PAD_))
        self.assertTrue(all(T.D['bonds']['c0']==tp._PAD_))
        self.assertTrue(all(T.D['angles']['c1']==tp._PAD_))
        self.assertTrue(all(T.D['angles']['c0']==tp._PAD_))
        for c in [f'c{i}' for i in range(6)]:
            self.assertTrue(all(T.D['dihedrals'][c]==tp._PAD_))
    def test_write_top(self):
        fn='test.top'
        T=tp.Topology.read_top(fn)
        # write into a temp dir: the conftest change_test_dir fixture puts cwd
        # inside the repo, so writing here leaves an artifact behind whenever
        # the test fails or is interrupted before its cleanup runs
        with tempfile.TemporaryDirectory() as td:
            out=os.path.join(td,'write_test.top')
            T.write_top(out)
            self.assertTrue(os.path.exists(out))
            W=tp.Topology.read_top(out,pad=pd.NA)
        self.assertTrue(all(W.D['bonds']['c1'].isna()))
        self.assertTrue(all(W.D['bonds']['c0'].isna()))
        self.assertTrue(all(W.D['angles']['c1'].isna()))
        self.assertTrue(all(W.D['angles']['c0'].isna()))
        for c in [f'c{i}' for i in range(6)]:
            self.assertTrue(all(W.D['dihedrals'][c].isna()))
        self.assertTrue(W.D['atoms'].shape==T.D['atoms'].shape)

