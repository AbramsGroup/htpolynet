"""

.. module:: test_resources
   :synopsis: tests resources
   
.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import unittest
import os
import logging
logger=logging.getLogger(__name__)
from htpolynet.utils.projectfilesystem import RuntimeLibrary
from htpolynet.external.software import Software

class TestResources(unittest.TestCase):
    def test_runtime_library_system(self):
        s=RuntimeLibrary.system()
        self.assertTrue(os.path.exists(s.root))
        self.assertTrue('cfg' in s.ResourcePaths)
        self.assertTrue('example_depot' in s.ResourcePaths)
        self.assertTrue('mdp' in s.ResourcePaths)
        self.assertTrue('molecules' in s.ResourcePaths)
        self.assertTrue('tcl' in s.ResourcePaths)
