.. _liquid_styrene_monomer:

The Styrene Monomer
-------------------

Styrene is a vinyl monomer; in this example we represent its **active**
form (ethylbenzene-like, all single bonds) rather than its natural
vinyl form (``c1ccccc1C=C``).  The active form is the template
``htpolynet`` uses for parameterization: the two vinyl carbons each
carry one *sacrificial* hydrogen, which is the H that a cure reaction
would later strip when forming a new C–C bond.  Since this example does
no curing, the sacrificial Hs simply stay put — but naming the carbons
now means the same monomer file can be reused in
:ref:`example 1 <ps_tutorial>` and beyond.

.. figure:: ../1-polystyrene/pics/STY.png
   :align: center
   :width: 240px

   Active styrene.  ``C1`` and ``C2`` are the (eventually) reactive
   vinyl carbons; each carries a sacrificial hydrogen.

In-config SMILES generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``constituents`` block of ``0-liquid-styrene.yaml`` carries the
SMILES string and, via the SMILES atom-mapping syntax (``[CH2:1]``,
``[CH3:2]``), the names of the two carbons that polymerization
reactions would target:

.. code-block:: yaml

   constituents:
     STY:
       smiles: "c1ccccc1[CH2:1][CH3:2]"
       reactive_atoms: {1: C1, 2: C2}
       count: 1000

When ``htpolynet run`` starts, it sees the ``smiles`` key, invokes
RDKit to generate ``lib/molecules/inputs/STY.mol2`` with the atom-name
mapping (``:1`` → ``C1``, ``:2`` → ``C2``), and then proceeds with the
normal AmberTools parameterization workflow (``antechamber`` →
``parmchk2`` → ``tleap``).  Users without RDKit can instead supply a
pre-built mol2 in ``lib/molecules/inputs/`` and use the
``rename_atoms`` key to assign atom names — see
:ref:`molecular_structure_inputs`.
