.. _ps_monomer:

Monomer: Styrene
----------------

.. _fig_sty_polym:

.. figure:: pics/styrene-polymerization.png

    Polymerization of styrene via carbon–carbon double-bond opening.
    The two reactive carbons we will name ``C1`` (the internal,
    phenyl-bearing carbon) and ``C2`` (the terminal methylene carbon).

`Styrene <https://pubchem.ncbi.nlm.nih.gov/compound/styrene>`_ polymerizes
by radical addition: the C=C double bond opens so that each radical
carbon bonds to a radical carbon on an adjacent monomer
(:numref:`fig_sty_polym`).  ``htpolynet`` needs to be able to refer to
those two carbons by name in the ``reactions`` block, so they get
explicit labels (``C1`` and ``C2``); the remaining atoms keep generic
names.

Valence-conservation and the active form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As described in the :ref:`user guide <molecular_structure_inputs>`,
``htpolynet`` requires each reactive atom to carry a *sacrificial
hydrogen* that is deleted when a new bond forms.  The natural,
"inactive" form of styrene has a C=C double bond and so the would-be
reactive carbons don't have an H to spare:

.. figure:: pics/STYCC.png
   :scale: 50 %

   Styrene (inactive form).

The **active** form we hand to ``htpolynet`` is ethylbenzene, which
carries two extra hydrogens on what will become the reactive carbons:

.. figure:: pics/STY.png
   :scale: 50 %

   Ethylbenzene (active form of styrene, monomer name ``STY``).

In-config SMILES generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``constituents`` block of ``1-polystyrene.yaml`` carries the SMILES
string and, via the SMILES atom-mapping syntax (``[CH2:1]``,
``[CH3:2]``), the names of the two carbons referenced by the cure and
cap reactions:

.. code-block:: yaml

   constituents:
     STY:
       smiles: "c1ccccc1[CH2:1][CH3:2]"
       reactive_atoms: {1: C1, 2: C2}
       count: 1000

When ``htpolynet run`` starts, it sees the ``smiles`` key, invokes
RDKit to generate ``lib/molecules/inputs/STY.mol2`` with atom names
``C1`` and ``C2`` substituted at the mapped positions, then proceeds
with the usual AmberTools parameterization
(``antechamber`` → ``parmchk2`` → ``tleap``).  The mol2 it writes
looks something like this, with the two reactive carbons named:

.. code-block:: text

   @<TRIPOS>ATOM
         1 C       ...  C.ar  1  STY  ...
         2 C       ...  C.ar  1  STY  ...
         3 C       ...  C.ar  1  STY  ...
         4 C       ...  C.ar  1  STY  ...
         5 C       ...  C.ar  1  STY  ...
         6 C       ...  C.ar  1  STY  ...
         7 C1      ...  C.3   1  STY  ...
         8 C2      ...  C.3   1  STY  ...
         9 H       ...  H     1  STY  ...
        ...

Users without RDKit can instead supply a pre-built mol2 in
``lib/molecules/inputs/`` and use the ``rename_atoms`` key in the
config to assign atom names by index — see
:ref:`molecular_structure_inputs` for the full alternative path.

The next page covers the :ref:`reaction dictionaries <ps_reactions>`
that tell ``htpolynet`` how to glue these monomers together.
