"""Manages bidirectional interatomic bondlists.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""

import numpy as np
import networkx as nx
import pandas as pd

import logging
logger=logging.getLogger(__name__)

class Bondlist:
    """ The member "B" is a dictionary keyed on atom index whose values of lists 
    of atom indices indicating bond partners of the key atom
    """
    def __init__(self):
        self.B={}
    
    @classmethod
    def fromDataFrame(cls,df:pd.DataFrame):
        inst=cls()
        inst.update(df)
        return inst

    def update(self,df:pd.DataFrame):
        """Updates the bondlist using data in the parameter dataframe df.

        Args:
            df (pd.DataFrame): dataframe with minimally columns named 'ai' and 'aj'

        Raises:
            Exception: if no column 'ai' or 'aj' in df
        """
        if not 'ai' in df.columns and not 'aj' in df.columns:
            raise Exception('Bondlist expects a dataframe with columns "ai" and "aj".')
        assert df['ai'].dtype==int
        assert df['aj'].dtype==int
        aiset=set(df.ai)
        ajset=set(df.aj)
        keyset=aiset.union(ajset)
        keys=sorted(list(keyset))
        self.B.update({k:[] for k in keys})
        assert all([type(x)==int for x in self.B.keys()])
        for r in df.itertuples():
            ai=r.ai 
            aj=r.aj
            self.B[ai].append(aj)
            self.B[aj].append(ai)
        for k,v in self.B.items():
            this_check=all([type(x)==int for x in v])
            if not this_check:
                logger.error(f'{k} {v}')
            assert this_check

    def __str__(self):
        retstr=''
        for k,v in self.B.items():
            retstr+=f'{k}: '+' '.join(str(vv) for vv in v)+'\n'
        return retstr
    
    def partners_of(self,idx):
        """Returns a copy of the value of self.B[idx].

        Args:
            idx (int): atom index

        Returns:
            list: list of indices of atoms to which atom 'idx' is bound
        """
        if idx in self.B:
            assert all([type(x)==int for x in self.B[idx]])
            return self.B[idx][:]
        return []

    def are_bonded(self,idx,jdx):
        """Returns True if atoms with indices idx and jdx are bonded neighbors.

        Args:
            idx (int): atom index
            jdx (int): another atom index

        Returns:
            bool: True if idx and jdx are bonded neighbors
        """
        if idx in self.B and jdx in self.B:
            return jdx in self.B[idx]
        return False

    def append(self,pair):
        """Appends the bonded pair in parameter 'pair' to the bondlist.

        Args:
            pair (list-like container): pair atom indices
        """
        assert len(pair)==2
        ai,aj=min(pair),max(pair)
        assert type(ai)==int
        assert type(aj)==int
        if not ai in self.B:
            self.B[ai]=[]
        self.B[ai].append(aj)
        if not aj in self.B:
            self.B[aj]=[]
        self.B[aj].append(ai)

    def delete_atoms(self,idx):
        """Deletes all instances of atoms in the list idx from the bondlist.

        Args:
            idx (list): list of indices for atoms to delete
        """
        ''' delete entries '''
        for i in idx:
            if i in self.B:
                del self.B[i]
        ''' delete members in other entries '''
        for k,v in self.B.items():
            for i in idx:
                if i in v:
                    self.B[k].remove(i)
    
    def adjacency_matrix(self):
        """Generates and returns an adjacency matrix built from the bondlist.

        Returns:
            numpy.ndarray: adjacency matrix
        """
        N=len(self.B)
        A=np.zeros((N,N)).astype(int)
        for i,n in self.B.items():
            for j in n:
                A[i-1,j-1]=1
                A[j-1,i-1]=1
        return A

    def as_list(self,root,depth):
        """Recursively builds a list of all atoms that form a bonded cluster by traversing maximally depth bonds.

        Args:
            root (list-like container of two ints): root bond
            depth (int): number of bonds to traverse to define bonded cluster

        Returns:
            list: list of atom indices in the bonded cluster
        """
        if depth==0:
            return [root]
        a,b=root
        abranch=[]
        for an in self.partners_of(a):
            if an!=b:
                abranch.extend(self.as_list([a,an],depth-1))
        bbranch=[]
        for bn in self.partners_of(b):
            if bn!=a:
                bbranch.extend(self.as_list([b,bn],depth-1))
        result=[root]
        result.extend(abranch)
        result.extend(bbranch)
        # sort entries, remove duplicates
        for i in range(len(result)):
            if result[i][0]>result[i][1]:
                result[i]=result[i][::-1]
        return result

    def half_as_list(self,root,depth):
        """Returns bonded cluster defined by atom b in root found by traversing depth bonds excluding the bond to atom a in root.

        Args:
            root (list-like container of two ints): root bond
            depth (int): number of bonds to traverse

        Returns:
            list: list of atom indices in bonded cluster "owned" by atom b not containing atom a
        """
        a,b=root
        self.delete_atoms([a])
        bbranch=[root]
        for bn in self.partners_of(b):
            if bn!=a:
                bbranch.extend(self.as_list([b,bn],depth-1))
        for i in range(len(bbranch)):
            if bbranch[i][0]>bbranch[i][1]:
                bbranch[i]=bbranch[i][::-1]
        bbranch.sort()
        red=[]
        for b in bbranch:
            if not b in red:
                red.append(b)
        return red    

    def graph(self):
        """Generates a networkx Graph object from the bondlist.

        Returns:
            networkx.Graph: a networkx Graph object
        """
        g=nx.Graph()
        for i,n in self.B.items():
            for j in n:
                g.add_edge(i,j)
        return g
