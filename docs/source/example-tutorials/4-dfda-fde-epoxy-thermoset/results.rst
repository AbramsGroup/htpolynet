.. _dfe_results:

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

   Density vs. time during densification of the DFA/FDE liquid.
   With ``initial_density: 300 kg/m³`` and a single 300 ps NPT
   segment the system reaches roughly 1.05 g/cm³ on the first
   pass — comparable to the DGEBA/PACM system, since the
   furan + cyclohexyl backbones pack at similar density.

.. figure:: pics/cure_info.png

   Left: cure conversion vs. wall-clock.  Right: cure iteration
   index vs. wall-clock.  Converges in 15 iterations to 95 %
   conversion plus a capping iteration that re-forms 20
   unreacted oxirane rings (see the run page).  The shape is
   typical of the DGEBA/PACM-style stepwise amine cure — long
   tail driven by the secondary-to-tertiary reaction having
   ``probability: 0.5`` and the late-stage pairs being scarce.

.. figure:: pics/reaction_network.png

   The user-declared reaction set as a bipartite DAG.  Two
   cure reactions (``Primary-to-secondary-amine`` and
   ``Secondary-to-tertiary-amine``) feed one intermediate
   (``DFA~N1-C1~FDE``) and one final crosslink product
   (``DFA~N1-C1~FDE-C1~FDE``); the cap reaction
   (``Oxirane-formation``) re-forms unreacted oxirane rings
   into ``FDEC``.

For end-to-end traces:

.. code-block:: console

   $ htpolynet plots build --proj proj-0 --buildplot t --traces t d p

.. figure:: pics/buildtraces.png

   Top: temperature vs. time (cumulative bond count overlaid).
   Middle: density vs. time.  Bottom: potential energy vs.
   time.  The first big temperature spike around 500 ps is the
   precure anneal cycle; each subsequent narrow temperature
   pulse with a paired density dip is one CURE iteration's
   relax + equilibrate cascade.  Density rises monotonically
   from ~0.3 g/cm³ at the end of densification to ~1.0 g/cm³
   after postcure, with the trace stepping up at each iteration
   boundary as new C-N bonds shrink the box.

Before and after
^^^^^^^^^^^^^^^^

Snapshots of the densified liquid vs. the cured network.  FDE
residues in mauve, DFA residues in green, capped (re-closed)
``FDEC`` oxirane rings in purple:

.. list-table::

    * - .. figure:: pics/dfa-fde-liq.png

           System before cure: densified liquid of FDE and DFA.

      - .. figure:: pics/dfa-fde-cured.png

           System after cure + cap: amine-bridged network with
           leftover oxiranes re-closed by the cap reaction.

Residue census
^^^^^^^^^^^^^^

On the representative run logged below (95 % cure, 380 / 400
C–N bonds, plus 20 cap-stage re-closures):

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Residue
     - Atom count
     - Source
   * - ``FDE``
     - 6380
     - 200 FDE × ~32 atoms each, the diepoxide.
   * - ``DFA``
     - 2520
     - 100 DFA × ~25 atoms each, the diamine.
   * - **Total**
     - **8900**
     - 100 DFA + 200 FDE = 300 monomers.

Profile
^^^^^^^

From a representative run:

.. code-block:: text

   Stage                                                   wall      subprocess
   ------------------------------------------------------------------------------
   setup                                                 2.06 s            0 ms
   initialization                                         <1 s             <1 s
   densification                                          ~10 s            ~10 s
   precure                                              2m36s           2m35s
   cure                                                33m03s              0 ms
     capping                                             1m07s         1m00s
   postcure                                             1m31s           1m31s
   final                                                3.78 s              0 s

Total: ~38 minutes — about the same as the PACM/DGEBA system (~50
min for a similar number of bonds) and orders of magnitude shorter
than the HTPB/IPDI system (~12 hours).  As with the other amine-cure examples,
``gmx-mdrun`` dominates subprocess time; ``setup`` is fast
because the diamine + diepoxide have only a few
parameterization templates compared to the procession-built
HTPB chains.

Try it
^^^^^^

A useful exercise on this system: run it twice, once with
``CURE.controls.min_bonds_per_iteration: 1`` and once with the
default ``min_bonds_per_iteration: 10``, and compare:

* total wall time;
* total number of CURE iterations;
* the breakdown of ``cure/iter-*`` wall times in
  ``profile.json``.

Because the late-iteration "one bond per iteration" regime is
particularly long-tailed in this system, this is a good
showcase for why the default isn't ``1``.
