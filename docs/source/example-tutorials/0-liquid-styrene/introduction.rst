.. _liquid_styrene_introduction:

Introduction
------------

This is the simplest possible ``htpolynet run``: parameterize one small
molecule (styrene), pack copies into a box, densify, and run a short
anneal — no cure reactions, no capping, no iteration.  It is a useful
shakedown of a fresh install: if this build completes cleanly, the
``htpolynet`` ↔ AmberTools ↔ GROMACS plumbing is healthy and you can
move on to the polymerizing examples.

It is also a reasonable way to produce an equilibrated liquid of a
monomeric species — for example, as a precursor configuration for later
cure runs, or as a starting point for small-molecule property estimates
(density, free volume, etc.).

Set up a clean working directory and pull the example YAML:

.. code-block:: console

   $ mkdir my_liquid_styrene
   $ cd my_liquid_styrene
   $ htpolynet fetch-example 0
   Fetched 0-liquid-styrene.yaml  (run with: htpolynet run 0-liquid-styrene.yaml)
   $ ls
   0-liquid-styrene.yaml

That single YAML is everything you need: it carries the styrene SMILES
inline, so ``htpolynet`` materializes ``STY.mol2`` for itself at the
start of the run.  No shell script, no pre-generated input files.
