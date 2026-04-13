"""Some convenient tools for handling pandas dataframes in the context of htpolynet coordinates.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import logging

import pandas as pd

logger=logging.getLogger(__name__)


def get_row_as_string(df:pd.DataFrame,attributes):
    """Returns the selected rows as a string, with rows expected to be uniquely defined by attributes dict.

    Args:
        df (pd.DataFrame): a pandas dataframe
        attributes (dict(str,obj)): dictionary of column names (keys) and values that specify set of rows to be returned

    Returns:
        str: selected dataframe converted to a string
    """
    ga={k:v for k,v in attributes.items() if k in df}
    c=[df[k] for k in ga]
    V=pd.Series(list(ga.values()))
    l=pd.Series([True]*df.shape[0])
    for i in range(len(c)):
        l = (l) & (c[i]==V[i])
    return df[list(l)].to_string()

def get_rows_w_attribute(df:pd.DataFrame,name,attributes:dict):
    """Returns a series of values of attribute "name" from all rows matching attributes dict.

    Returns:
        values: list of values from selected rows
    """
    ga={k:v for k,v in attributes.items() if k in df}
    assert len(ga)>0,f'Cannot find any rows with attributes {attributes}'
    if type(name)==list:
        name_in_df=all([n in df for n in name])
    else:
        name_in_df= name in df
    assert name_in_df,f'Attribute(s) {name} not found'
    c=[df[k] for k in ga]
    V=pd.Series(list(ga.values()))
    l=pd.Series([True]*df.shape[0])
    for i in range(len(c)):
        l = (l) & (c[i]==V[i])
    return df[list(l)][name].values

def set_row_attribute(df:pd.DataFrame,name,value,attributes):
    """Sets value of attribute name to value in all rows matching attributes dict.

    Args:
        df (pd.DataFrame): a pandas dataframe
        name (str): name of attribute whose value is to be set
        value (scalar): value the attribute is to be set to
        attributes (dict): dictionary of attribute:value pairs that specify the atoms whose attribute is to be set
    """
    ga={k:v for k,v in attributes.items() if k in df}
    exla={k:v for k,v in attributes.items() if not k in df}
    if len(exla)>0:
        logger.warning(f'Caller attempts to use unrecognized attributes to refer to row: {exla}')
    if name in df and len(ga)>0:
        c=[df[k] for k in ga]
        V=pd.Series(list(ga.values()))
        l=pd.Series([True]*df.shape[0])
        for i in range(len(c)):
            l = (l) & (c[i]==V[i])
        cidx=[c==name for c in df.columns]
        df.loc[list(l),cidx]=value

def set_rows_attributes_from_dict(df:pd.DataFrame,valdict,attributes):
    """Sets values of attributes in valdict dict of all rows matching attributes dict.

    Args:
        df (pd.DataFrame): a pandas dataframe
        valdict (dict): dictionary of attribute:value pairs to set
        attributes (dict): dictionary of attribute:value pairs that specify the atoms whose attribute is to be set
    """
    ga={k:v for k,v in attributes.items() if k in df}
    exla={k:v for k,v in attributes.items() if not k in df}
    if len(exla)>0:
        logger.warning(f'using unknown attributes to refer to atom: {exla}')
    if all([x in df for x in valdict]) and len(ga)>0:
        c=[df[k] for k in ga]
        V=pd.Series(list(ga.values()))
        l=pd.Series([True]*df.shape[0])
        for i in range(len(c)):
            l = (l) & (c[i]==V[i])
        for k,v in valdict.items():
            cidx=[c==k for c in df.columns]
            df.loc[list(l),cidx]=v 
