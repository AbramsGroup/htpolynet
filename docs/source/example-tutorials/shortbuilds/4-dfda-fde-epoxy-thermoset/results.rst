.. _dfe_results:

Results
-------

The standard final-results bundle lives in
``proj-0/systems/final-results/``:

.. code-block:: console

   $ vmd final.viz.psf final.gro -e final.viz.tcl

Diagnostic-log plots:

.. code-block:: console

   $ htpolynet plots diag --diags diagnostics.log

The output figures land in ``proj-0/plots/``.  Compared to the
DGEBA/PACM run, expect:

* A similar overall conversion-vs-wall-clock shape — the late
  iterations dominate the tail.
* A slightly higher ``setup`` share in ``profile.json`` because of the
  aromatic-heterocycle type-perception overhead.
* A network density in the same ballpark as DGEBA/PACM (~1.0–1.1 g/cm³
  after postcure), though the exact number depends on the run's
  equilibration discipline.

For full system-build traces:

.. code-block:: console

   $ htpolynet plots build --proj proj-0 --buildplot t --traces t d p

Try it
^^^^^^

A useful exercise on this system: run it twice, once with
``CURE.controls.min_bonds_per_iteration: 1`` and once with the
default ``min_bonds_per_iteration: 10``, and compare:

* total wall time;
* total number of CURE iterations;
* the breakdown of ``cure/iter-*`` wall times in ``profile.json``.

Because the late-iteration "one bond per iteration" regime is
particularly long-tailed in this system, this is a good showcase for
why the default isn't ``1``.

A note on figures
^^^^^^^^^^^^^^^^^

This tutorial doesn't yet ship the before/after molecular snapshots
that examples 2 and 3 do.  If you run the example and produce
informative VMD renders or build-trace plots, drop them into
``docs/source/example-tutorials/shortbuilds/4-dfda-fde-epoxy-thermoset/pics/``
and add ``.. figure::`` blocks where they fit.
