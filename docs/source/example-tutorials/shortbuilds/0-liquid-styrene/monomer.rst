.. _liquid_styrene_monomer:

The Styrene Monomer
-------------------

Styrene is a vinyl monomer; in this example we represent its **active**
form (saturated, ethylbenzene-like) so that the reactive vinyl carbons
each carry one sacrificial hydrogen.  Because we do no polymerization
here, that "activeness" only matters insofar as it gives us a single
all-single-bonds structure to densify.

.. todo::

   - Insert an image of the active styrene structure (e.g. reuse
     ``../1-polystyrene/pics/STY.png``).
   - Explain why the SMILES used is ``c1ccccc1[CH2:1][CH3:2]`` rather
     than the "natural" ``c1ccccc1C=C``: the saturated form is the
     valence-conserving template (vinyl carbons each have a sacrificial
     hydrogen).

In-config SMILES generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``constituents`` block of ``0-liquid-styrene.yaml`` carries the
SMILES string and the names of the two atoms that would *eventually* be
the polymerization sites (here labelled with the SMILES atom-mapping
syntax ``[CH2:1]`` and ``[CH3:2]``):

.. code-block:: yaml

   constituents:
     STY:
       smiles: "c1ccccc1[CH2:1][CH3:2]"
       reactive_atoms: {1: C1, 2: C2}
       count: 200

When ``htpolynet run`` starts, it sees the ``smiles`` key, invokes RDKit
(or falls back to ``obabel`` with an explicit index map — see
:ref:`molecular_structure_inputs`) to generate
``lib/molecules/inputs/STY.mol2``, and then proceeds with the normal
parameterization workflow.  Even though there are no reactions for
``htpolynet`` to apply to those atoms in this example, naming them now
means the file can be reused as a drop-in monomer template in
:ref:`example 1 <ps_tutorial>` and beyond.

.. todo::

   - Add a snippet of the generated ``STY.mol2`` showing the atom block
     with C1 / C2 in place.
   - Note the ``rename_atoms`` alternative for users without RDKit.
