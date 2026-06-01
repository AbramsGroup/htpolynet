.. _badcy_results:

Results
-------

The standard final-results bundle is in
``proj-0/systems/final-results/``:

.. code-block:: console

   $ vmd final.viz.psf final.gro -e final.viz.tcl

Diagnostic-log plots:

.. code-block:: console

   $ htpolynet plots diag --diags diagnostics.log

.. figure:: pics/densification-density.png

   Density vs. time during densification of the BPA + 1,3,5-triazine
   liquid.  With ``initial_density: 200 kg/m³`` the system reaches
   roughly 0.9-1.0 g/cm³ after the four 100 ps NPT segments — a touch
   below typical polymer density because the unreacted starting state
   carries free triazine and free BPA-OH that don't pack tightly.

.. figure:: pics/cure_info.png

   Left: cure conversion vs. wall-clock.  Right: cure iteration index
   vs. wall-clock.  90 % conversion in 9 iterations; the long-tail
   shape is sharper than the DGEBA/PACM example because the A2+B3
   step-growth on a 3:2 stoichiometry runs out of nearby reactive
   pairs faster than the diepoxide-amine system does.

.. figure:: pics/reaction_network.png

   The final bond network as a graph.  Triazine residues that ended
   up fully bonded (k=3) appear as 3-degree nodes; ones that ended up
   with k<3 are the ones the postcure repair stage subsequently
   dismantled.

For end-to-end traces:

.. code-block:: console

   $ htpolynet plots build --proj proj-0 --buildplot t --traces t d p

.. figure:: pics/buildtraces.png

   Top: temperature vs. time across the full build (cumulative bond
   count overlaid).  Middle: density vs. time.  Bottom: potential
   energy vs. time.  The flat-temperature postcure-anneal segments
   between 300 K and 500 K dominate the right end of the trace.

Before and after
^^^^^^^^^^^^^^^^

.. admonition:: Placeholder
   :class: caution

   **TODO:** snapshots of the densified liquid vs. the cured +
   repaired network.  Render via ``vmd final.viz.psf final.gro -e
   final.viz.tcl`` on each of ``proj-0/systems/densification/`` and
   ``proj-0/systems/final-results/``, colour BPA mauve, intact
   ``TAZ`` cyan, and ``CYN`` red.  Save as
   ``pics/badcy-liq.png`` and ``pics/badcy-cured.png``.

Residue census
^^^^^^^^^^^^^^

On the representative run logged above (90 % cure, 63 incomplete
triazines, 117 in-place caps + 72 free caps):

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Residue
     - Count
     - Source
   * - ``BPA``
     - 360
     - All 360 input BPAs survive (no monomer-level deletion).
   * - ``TAZ``
     - 177
     - The 240 - 63 = 177 triazines that reached k=3 during cure.
   * - ``CYN``
     - 189
     - 63 dismantled rings × 3 -C#N fragments = 189 cap residues.

Atom conservation is exact: every C and N atom of every dismantled
triazine ends up in exactly one ``CYN`` residue, and the only atoms
deleted are the (3-k) × 2 sacrificial H atoms per dismantled ring (one
per dangling ring C, plus one per donated free cap's BPA-O target).

Profile
^^^^^^^

The end-of-run profile (in ``console.log`` and
``proj-0/profile.json``) shows where the wall time went.  In a typical
BADCy run, expect:

* ``setup`` is fast (only 11 templates to parameterize, all small
  molecules);
* ``precure`` is a meaningful share (~3-4 min) because of the long
  preequilibration + anneal block;
* ``cure`` dominates (~75 % of wall time) — same shape as any
  step-growth example;
* ``repair`` adds about a minute: the topology surgery itself is fast
  (~5 s), the rest is the post-surgery minimization + 5 ps NVT
  settle;
* ``postcure`` is one anneal cycle + a 100 ps NPT.

A useful diagnostic: ``grep "incomplete TAZ" diagnostics.log`` reports
the count of dismantled triazines and the in-place / free-fragment
split, which together with the residue census above tell you how
"complete" the topological network was just before the repair stage
ran.

The next page covers the same kind of postsim + analyze workflow
documented for :ref:`tutorial 3 <pde_postsim>`.
