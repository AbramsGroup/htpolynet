.. _bgs_introduction:

Introduction
------------

Set up a clean working directory and pull the example YAML:

.. code-block:: console

   $ mkdir my_bisgma_styrene
   $ cd my_bisgma_styrene
   $ htpolynet fetch-example 2
   Fetched 2-bisgma-styrene-thermoset.yaml  (run with: htpolynet run 2-bisgma-styrene-thermoset.yaml)
   $ ls
   2-bisgma-styrene-thermoset.yaml

As with examples 0 and 1, this is a single self-contained YAML — no shell
script, no pre-built ``mol2`` files.  The three primitive monomers (STY,
BPA, HIE) are each declared with a SMILES string and atom-mapping tokens
in ``constituents``, and ``htpolynet`` materializes their ``mol2`` files
at the start of the run.

What is different here is that **bisGMA itself is not a primitive monomer**
in the config — we declare a fourth species ``GMA`` with no SMILES, and a
pair of ``param``-stage reactions that tell ``htpolynet`` how to build it
by joining one HIE onto each phenolic hydroxyl of BPA.  The build pipeline
then proceeds with STY and GMA as the actual constituents of the initial
liquid, and runs the cure between STY/HIE radicals.

The :ref:`next page <bgs_reactions>` walks through why we do it this way
(rather than describing GMA directly) and what the cure and cap reaction
set looks like.
