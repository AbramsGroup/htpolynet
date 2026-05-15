.. _ps_introduction:

Introduction
------------

Set up a clean working directory and pull the example YAML:

.. code-block:: console

   $ mkdir my_polystyrene
   $ cd my_polystyrene
   $ htpolynet fetch-example 1
   Fetched 1-polystyrene.yaml  (run with: htpolynet run 1-polystyrene.yaml)
   $ ls
   1-polystyrene.yaml

Like example 0, this is a single self-contained YAML — no shell script,
no pre-generated mol2.  The styrene SMILES is in the ``constituents``
block and ``htpolynet`` materializes ``STY.mol2`` itself at the start of
the run.  What this example *adds* over example 0 is the cure chemistry:
a single C–C inter-monomer reaction (``cure`` stage) and a corresponding
cap reaction (``cap`` stage) that restores the vinyl double bond on
any monomer that did not react.

.. note::

   With ``count: 1000`` and a full cure loop, this is a noticeably
   longer build than example 0 (same monomer count, but the cure
   iterations dominate overall wall time).  If you just want a smoke
   test, drop ``constituents.STY.count`` to ~200 before running.

The next page covers what the styrene "monomer" actually looks like in
the config, and why we represent it in its saturated *active* form
rather than as the natural vinyl.
