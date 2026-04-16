.. _ps_reactions:

Reactions
=========

**Cure reaction**

The only inter-monomer reaction is the bond between ``C1`` of one styrene and ``C2`` of another:

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

The ``product`` name ``STY~C1-C2~STY`` is used internally by ``htpolynet`` to build and parameterize the dimer template.  This template provides atom types, charges, and bonded interactions that are mapped back onto the growing polymer whenever a new C1–C2 bond is formed.

**Chain expansion**

Because styrene polymerizes by C=C double-bond opening, the reactive atoms ``C1`` and ``C2`` can form chains of arbitrary length through the polymer backbone.  ``htpolynet`` automatically expands the base reaction above into all combinations needed to correctly parameterize trimer and tetramer templates.  These cover the full set of dihedral topologies that arise as chains grow.  You do not need to list these explicitly; ``htpolynet`` reports them in the console log at the start of a run.

**Cap reaction**

Once CURE finishes, any monomer that did not react still carries its two sacrificial hydrogens and its C1–C2 atoms remain in the "active" (saturated) form.  The cap reaction converts these back to the true C=C double bond:

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

The ``stage: cap`` label means this reaction is applied only after CURE has completed.

The next thing we consider is the :ref:`configuration file <ps_configuration>`.
