.. _bgs_results:

Results
-------

At the end of the run, ``proj-0/systems/final-results/`` contains the
usual five files (``final.gro``, ``final.top``, ``final.tpx``,
``final.grx``, plus the ``final.viz.psf``/``final.viz.tcl`` pair and a
``final.viz.macros.tcl`` of constituent-keyed selection macros).
Open with:

.. code-block:: console

   $ vmd final.viz.psf final.gro -e final.viz.tcl

Because bis-GMA is a *built* constituent (``BPA`` + 2 ``HIE`` joined by
the two param-stage reactions in the YAML), each GMA molecule appears
in the system gro as three residues with their building-block names.
The constituent macros loaded from ``final.viz.macros.tcl`` let you
select whole bis-GMAs as one chemical entity — useful for highlighting
the network in the cured snapshot below:

.. code-block:: tcl

   mol modselect 0 top GMA       ;# all 75 bis-GMA molecules
   mol modselect 0 top STY       ;# all 150 styrenes
   mol modselect 0 top GMA_007   ;# one specific bis-GMA (global molecule index 7)

Plots from the build log
^^^^^^^^^^^^^^^^^^^^^^^^

Diagnostic-log plots:

.. code-block:: console

   $ htpolynet plots diag --diags diagnostics.log

.. figure:: pics/densification-density.png

   Density vs. time during the densification stage of the bisGMA/STY
   liquid.  Each of the eight 100 ps NPT segments is colored
   distinctly along the ``matplotlib`` ``plasma`` colormap.

.. figure:: pics/cure_info.png

   Cure progress vs. wall time and per-iteration bond yield.  Roughly
   75 % of the conversion is reached in the first 25 % of the cure
   wall time — the long tail belongs to the late iterations that pick
   up only a handful of bonds each.

For a full system-build trace from ``edr`` files:

.. code-block:: console

   $ htpolynet plots build --proj proj-0 --buildplot t --traces t d p

This walks every ``edr`` in the project tree and stitches together the
trace; it takes a few minutes on a moderate-sized build.

.. figure:: pics/buildtraces.png

   Top: temperature vs. time for the 95 % cure of bisGMA/STY.  Middle:
   density vs. time, overlaid with cumulative bond count.  Bottom:
   potential energy vs. time.

Before and after
^^^^^^^^^^^^^^^^

Snapshots of one complete crosslink site (three connected styrene
units lifted from the cured box) alongside the initial liquid and
the cured system.  Atoms are coloured CPK by element — carbon grey,
nitrogen blue, oxygen red — so the colours in the detail panel
transfer directly to the bulk views.  In the bulk panels the
``C1``-``C2`` chain carbons of STY and HIE residues are drawn in
thick CPK Licorice on a faded grey lines background, so the new
chain bonds formed during cure stand out from the carrier mass:

.. list-table::

    * - .. figure:: pics/gma-sty-detail.png

           One complete crosslink: three styrenes bonded into a chain
           segment via cure-formed ``C1``-``C2`` C-C bonds.

      - .. figure:: pics/gma-sty-liq.png

           Liquid before cure.  Each monomer's intra-monomer ``C1``-``C2``
           single bond is in licorice; cure will convert most of these
           into the network backbone.

      - .. figure:: pics/gma-sty-cured.png

           Cured system.  All ``C1``-``C2`` bonds (intra and inter)
           are now in licorice; the new C-C inter-monomer bonds are
           the cure crosslinks.

Profile
^^^^^^^

The end-of-run profile report (in ``console.log`` and
``proj-0/profile.json``) shows wall-time and aggregate subprocess time
by stage.  In a build like this the ``setup`` stage is non-trivial
because of all the chain-expansion templates that have to be
parameterized through AmberTools; expect ``antechamber`` and ``tleap``
to take a meaningful share of total wall time alongside ``gmx-mdrun``.