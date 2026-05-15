.. _ps_reactions:

Reactions
---------

Cure reaction
^^^^^^^^^^^^^

The single inter-monomer cure reaction forms a bond between ``C1`` of
one styrene and ``C2`` of another:

.. code-block:: yaml

   reactions:
     - name:        sty1_1
       stage:       cure
       reactants:
         1: STY
         2: STY
       product:     STY~C1-C2~STY
       probability: 1.0
       atoms:
         A:
           reactant: 1
           resid: 1
           atom: C1
           z: 1
         B:
           reactant: 2
           resid: 1
           atom: C2
           z: 1
       bonds:
         - atoms: [A, B]
           order: 1

The ``product`` name ``STY~C1-C2~STY`` is used internally by
``htpolynet`` to build and parameterize the dimer template.  That
template provides atom types, charges, and bonded interactions which
get mapped back onto the growing polymer every time a new ``C1–C2``
bond is formed during CURE.

Chain expansion
^^^^^^^^^^^^^^^

Because styrene polymerizes by C=C double-bond opening, ``C1`` and
``C2`` can extend into chains of arbitrary length through the polymer
backbone.  ``htpolynet`` automatically expands the dimer reaction above
into all trimer and tetramer combinations needed to correctly
parameterize the dihedral environments that arise as chains grow.  You
do not list these explicitly; ``htpolynet`` reports the expanded set in
the diagnostic log at the start of a run.

Cap reaction
^^^^^^^^^^^^

Once CURE finishes, any monomer that did not react still carries its
two sacrificial hydrogens and its ``C1–C2`` atoms are in the saturated
(ethylbenzene-like) "active" form.  The cap reaction converts these
back to the true ``C=C`` double bond:

.. code-block:: yaml

   - name:         styCC
     stage:        cap
     reactants:
       1: STY
     product:      STYCC
     probability:  1.0
     atoms:
       A:
         reactant: 1
         resid: 1
         atom: C1
         z: 1
       B:
         reactant: 1
         resid: 1
         atom: C2
         z: 1
     bonds:
       - atoms: [A, B]
         order: 2

The ``stage: cap`` label is what tells ``htpolynet`` to defer this
reaction until after CURE has converged; the bond ``order: 2`` is what
restores the double bond and strips the two sacrificial Hs.

The next page walks through the full
:ref:`configuration file <ps_configuration>`.
