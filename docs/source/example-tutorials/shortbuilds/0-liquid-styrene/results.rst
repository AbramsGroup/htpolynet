.. _liquid_styrene_results:

Results
-------

At the end of the run, ``proj-N/systems/final-results/`` contains:

* ``final.gro``, ``final.top``, ``final.tpx``, ``final.grx`` — the
  densified, post-anneal liquid in GROMACS-friendly form.
* ``final.viz.psf``, ``final.viz.tcl`` — VMD-friendly companion files.
  The PSF carries the real bond topology and the TCL helper trims
  bonds that wrap across the periodic image, so VMD does not draw the
  spurious long bonds you would otherwise see in a periodic box.  Open
  with:

  .. code-block:: console

     $ vmd final.viz.psf final.gro -e final.viz.tcl

A useful sanity check: read off the final box volume and the total
mass (1000 × styrene = 1000 × 104.15 g/mol) to compute density, and
compare to liquid styrene's literature value of ~909 kg/m³ at 25 °C.
Because the equilibration is deliberately short, expect the result to
*approach* but not match the literature value — the point of this
example is plumbing, not production-quality property estimation.

Post-build analyses
^^^^^^^^^^^^^^^^^^^

The comment block at the top of ``0-liquid-styrene.yaml`` sketches an
``htpolynet analyze`` configuration that runs three quick analyses on
the densification output: a trajectory ``check``, a methods report,
and a ``freevolume`` calculation.  Save those YAML lines into
``ck.yaml`` and run:

.. code-block:: console

   $ htpolynet analyze -cfg ck.yaml -ocfg 0-liquid-styrene.yaml -proj proj-0

Results land under ``proj-0/analyze/{check,report-methods,freevolume}/``.
See :ref:`tutorials_postsim_analyses` for a fuller worked example of
the analysis subsystem.
