.. _liquid_styrene_introduction:

Introduction
------------

Set up a clean working directory and pull the example YAML with
``htpolynet fetch-example``:

.. code-block:: console

   $ mkdir my_liquid_styrene
   $ cd my_liquid_styrene
   $ htpolynet fetch-example 0
   Fetched 0-liquid-styrene.yaml  (run with: htpolynet run 0-liquid-styrene.yaml)
   $ ls
   0-liquid-styrene.yaml

The example is a single self-contained YAML.  No shell script, no
pre-generated monomer files: ``htpolynet`` materializes ``STY.mol2`` from a
SMILES string embedded in the config (see :ref:`monomer
<liquid_styrene_monomer>`) at the start of the run.

.. todo::

   - Describe *why* one would want a liquid build as a starting point
     (precursor / equilibrated initial condition for later cure runs,
     small-molecule property estimation, etc.).
   - Mention the expected wall-clock for this example on a modest GPU vs CPU.
