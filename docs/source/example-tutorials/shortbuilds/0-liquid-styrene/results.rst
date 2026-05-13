.. _liquid_styrene_results:

Results
-------

At the end of the run, ``proj-N/systems/final-results/`` contains:

* ``final.gro``, ``final.top``, ``final.tpx``, ``final.grx`` — the
  densified liquid in gromacs-friendly form.
* ``final.viz.psf``, ``final.viz.tcl`` — VMD-friendly companion files;
  open with ``vmd final.viz.psf final.gro -e final.viz.tcl``.

.. todo::

   - Show a representative density-vs-time plot from the densification
     stage.
   - Report a typical final density and compare to the literature value
     for liquid styrene (~909 kg/m³ at room temperature).
   - Include a VMD snapshot of the densified box for visual sanity.
   - Optionally walk through the ``htpolynet analyze`` post-simulation
     analyses listed at the top of the YAML.
