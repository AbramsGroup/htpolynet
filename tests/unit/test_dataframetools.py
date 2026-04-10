"""

.. module:: test_dataframetools
   :synopsis: tests dataframetools
   
.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import unittest
import os
import logging
logger=logging.getLogger(__name__)
from htpolynet.utils.dataframetools import get_rows_w_attribute, set_row_attribute, set_rows_attributes_from_dict
import pandas as pd

class TestDataframeTools(unittest.TestCase):
    def test_get_rows_w_attribute(self):
        df=pd.DataFrame({
            'a':[ 1, 2, 3, 4, 5],
            'b':[ 6, 7, 8, 9,10],
            'c':[11,12,13,14,15]
        })
        result=get_rows_w_attribute(df,'c',{'a':3,'b':8})
        self.assertEqual(result[0],13)
