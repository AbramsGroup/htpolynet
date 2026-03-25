"""Manages bond templates (bonds defined by type) and reaction bonds (bonds defined by instances).

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import logging
from copy import deepcopy

class BondTemplate:
    def __init__(self,names,resnames,intraresidue,order,bystander_resnames,bystander_atomnames,oneaway_resnames,oneaway_atomnames):
        """Creates a BondTemplate object.

        Args:
            names (list-like container of two ints): names of atoms in the bond
            resnames (list-like container of two strs): names of the two residues to which the atoms belong
            intraresidue (bool): True if this is an intraresidue bond
            order (int): bond order (1=single, 2=double, ...)
            bystander_resnames (list of two list-like containers of strs): lists of names of bystander residues (residues also bound to one of the atoms), one for each atom in names
            bystander_atomnames (list of two list-like containers of strs): lists of names of atoms in bystander residues (parallel to bystander_resnames)
            oneaway_resnames (list-like container of strs): names of one-away residues (residues bound one bond away from the new interresidue bond; only relevant for C=C free-radical polymerization)
            oneaway_atomnames (list-like container of strs): names of atoms in one-away residues
        """
        self.names=names
        self.resnames=resnames
        self.intraresidue=intraresidue
        self.bystander_resnames=bystander_resnames
        self.bystander_atomnames=bystander_atomnames
        self.oneaway_resnames=oneaway_resnames
        self.oneaway_atomnames=oneaway_atomnames
        self.order=order
    def reverse(self):
        """Reverses the order of all parallel lists in a BondTemplate object."""
        self.names=self.names[::-1]
        self.resnames=self.resnames[::-1]
        self.bystander_resnames=self.bystander_resnames[::-1]
        self.bystander_atomnames=self.bystander_atomnames[::-1]
        self.oneaway_resnames=self.oneaway_resnames[::-1]
        self.oneaway_atomnames=self.oneaway_atomnames[::-1]
    def __str__(self):
        return f'BondTemplate {self.names} resnames {self.resnames} intraresidue? {self.intraresidue} order {self.order} bystander-resnames {self.bystander_resnames} bystander-atomnames {self.bystander_atomnames} oneaway-resnames {self.oneaway_resnames} oneaway-atomnames {self.oneaway_atomnames}'
    def __eq__(self,other):
        check=self.names==other.names
        check=check and self.intraresidue==other.intraresidue
        check=check and self.resnames==other.resnames
        check=check and self.bystander_resnames==other.bystander_resnames
        check=check and self.bystander_atomnames==other.bystander_atomnames
        check=check and self.oneaway_resnames==other.oneaway_resnames
        check=check and self.oneaway_atomnames==other.oneaway_atomnames
        return check
    def is_reverse_of(self,other):
        """Returns True if self and other are reverse of each other.

        Args:
            other (BondTemplate): another BondTemplate object

        Returns:
            bool: True if self and other are reverse copies of each other
        """
        rb=deepcopy(other)
        rb.reverse()
        return self==rb

BondTemplateList=list[BondTemplate]

class ReactionBond:
    def __init__(self,idx,resids,order,bystanders,bystanders_atomidx,oneaways,oneaways_atomidx):
        """Generates a new ReactionBond object.

        Args:
            idx (list-like container of two ints): indices of two atoms that form the bond
            resids (list-like container of two ints): resids of the two atoms that form the bond
            order (int): order of the bond (1=single, 2=double, ...)
            bystanders (list of two list-like containers of ints): two lists of indices of interresidue resids already bound to each atom in idx
            bystanders_atomidx (list of two list-like containers of ints): two lists of indices of interresidue atoms already bound to each atom in idx
            oneaways (list of ints): list of one-away resids
            oneaways_atomidx (list of ints): list of one-away atom indices
        """
        self.idx=idx
        self.resids=resids
        self.bystander_resids=bystanders
        self.bystander_atomidx=bystanders_atomidx
        self.oneaway_resids=oneaways
        self.oneaway_atomidx=oneaways_atomidx
        self.order=order
    def reverse(self):
        self.idx=self.idx[::-1]
        self.resids=self.resids[::-1]
        self.bystander_resids=self.bystander_resids[::-1]
        self.bystander_atomidx=self.bystander_atomidx[::-1]
        self.oneaway_resids=self.oneaway_resids[::-1]
        self.oneaway_atomidx=self.oneaway_atomidx[::-1]
    def __str__(self):
        return f'ReactionBond {self.idx} resids {self.resids} order {self.order} bystander-resids {self.bystander_resids} oneaway-resids {self.oneaway_resids} oneaway-atomidx {self.oneaway_atomidx}'

ReactionBondList=list[ReactionBond]
