.. _example_tutorials:

Example Tutorials
=================

We present here five tutorials to help illustrate usage of ``htpolynet``.  Each tutorial walks through one polymerization recipe end-to-end: setting up monomer structures and reactions, running the build with ``htpolynet run``, and then performing post-build MD simulations and analyses with ``htpolynet postsim``, ``plots``, and ``analyze``.  Example ``0`` (liquid styrene) does no polymerization and is the simplest possible build; the remaining four add cure and cap reactions of increasing complexity.  Each tutorial number matches the example number returned by ``htpolynet fetch-example``.

.. note::

   **IMPORTANT DISCLAIMER**

   These are *not* production-level builds.  The system sizes are *way too small* and the equilibration and post-build simulation times are *way too short*.  You as a user are responsible for conducting the appropriate finite-size-effect tests and equilibration tests needed to guarantee robustness of your simulations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   0-liquid-styrene/index
   1-polystyrene/index
   2-bisgma-styrene-thermoset/index
   3-pacm-dgeba-epoxy-thermoset/index
   4-dfda-fde-epoxy-thermoset/index
