.. _htpb_run:

Running the Build
-----------------

From inside the working directory containing ``5-htpb-ipdi.yaml``:

.. code-block:: console

   $ htpolynet run -diag diagnostics.log 5-htpb-ipdi.yaml &> console.log &

This is by far the longest of the depot examples — plan for **half a
day to overnight** rather than a coffee break.  The stage layout
under ``proj-N/systems/`` is the standard one (``init/``,
``densification/``, ``precure/``, ``iter-K/``, ``capping/``,
``postcure/``, ``final-results/``, plus ``plots/`` and
``profile.json``).

Setup
^^^^^

``htpolynet`` parameterizes the 35 templates discussed in the
:ref:`configuration page <htpb_configuration>`.  Among the
interesting ones:

.. code-block:: text

   INFO> 35 molecules detected in 5-htpb-ipdi.yaml
   INFO>                       explicit: 34
   INFO>     implied by stereochemistry: 1
   INFO>            implied by symmetry: 0
   INFO> OB: generating mol2 from SMILES via RDKit
   INFO> TB: generating mol2 from SMILES via RDKit
   INFO> TBO: generating mol2 from SMILES via RDKit
   INFO> IPD: generating mol2 from SMILES via RDKit
   INFO> AmberTools> generating GAFF parameters from OB.mol2
   ... (4 small constituents parameterized)
   INFO> AmberTools> generating GAFF parameters from A2.mol2
   ... (6 param-stage products parameterized)
   INFO> AmberTools> generating GAFF parameters from A18_I0.mol2
   INFO> AmberTools> generating GAFF parameters from A18_I1.mol2
   ... (16 procession-iteration A18 intermediates parameterized)
   INFO> AmberTools> generating GAFF parameters from DHT.mol2
   INFO> AmberTools> generating GAFF parameters from THT.mol2
   ... (final assembled chains)
   INFO> Generated 35 molecule templates
   INFO> Initial composition is IPD 125, DHT 50, THT 50
   INFO> 100% conversion is 250 bonds

Conformer generation runs next: 6 thermalized conformers of each
``DHT`` and ``THT`` chain at 900 K via short GROMACS NVT MD.  This
takes roughly 5 minutes per conformer (chain × 6 conformers × 2
chains = 12 conformer-generation MD runs), accounting for ~half of
the setup time.

Densification + precure
^^^^^^^^^^^^^^^^^^^^^^^

The 25 densification NPT repeats at 600 K / 10 bar progressively
compact a very dilute initial state into a near-melt density of
~0.9-1.0 g/cm³.  Each repeat is 100 ps; the full densification
takes ~50 minutes of wall clock.  Precure adds a 300 ps NPT
preequilibration at 300 K / 1 bar, then a long anneal cycle (two
cycles between 300 and 600 K, 500 ps per segment) so the chains can
explore conformational space before cure starts.  Total precure
wall-clock: ~2 hours.

Cure
^^^^

CURE runs until either ``desired_conversion: 0.95`` or
``max_iterations: 150`` is reached.  On a representative run cure
converges in **15 iterations**.  The per-iteration wall-times are
revealing:

.. list-table::
   :header-rows: 1
   :widths: 15 25 25 35

   * - Iteration
     - Bonds formed
     - Cumulative conversion
     - Wall time
   * - 1
     - 47
     - 0.188
     - 10:36
   * - 2
     - 25
     - 0.288
     - 8:58
   * - 3
     - 19
     - 0.364
     - 10:18
   * - 4
     - 12
     - 0.412
     - 8:55
   * - 5
     - 19
     - 0.488
     - 8:54
   * - 6
     - 12
     - 0.536
     - 10:01
   * - 7
     - 19
     - 0.612
     - 17:09
   * - 8
     - 11
     - 0.656
     - 14:48
   * - 9
     - 17
     - 0.724
     - 16:55
   * - 10
     - 11
     - 0.768
     - 19:40
   * - 11
     - 10
     - 0.808
     - 33:06
   * - 12
     - 10
     - 0.848
     - 32:45
   * - 13
     - 10
     - 0.888
     - 1:00:30
   * - 14
     - 11
     - 0.932
     - 2:17:33
   * - 15
     - 5
     - 0.952
     - 3:49:45

Look at the bottom of that table: iteration 15 alone took **nearly
4 hours**.  This is the cure-tail effect at its most extreme — by
the time only a handful of hydroxyl / isocyanate pairs are left
unbonded, finding pairs within the bond-search radius requires the
``cure_drag`` step to pull distant atoms together over multiple MD
segments.  ``min_bonds_per_iteration: 10`` is what keeps the
iteration count from blowing up to 50+ at the tail; raising it
further would slightly reduce iteration count but each iteration
would have to drag further-apart atoms together, with diminishing
returns.

Total cure wall-time: ~10.3 hours.  Capping is trivially fast (0
bonds — all reactive sites that were going to bond did) and runs in
milliseconds.

Postcure
^^^^^^^^

Postcure runs two anneal cycles between 300 K and 600 K (50 ps per
segment) followed by a 1 ns NPT postequilibration at 300 K / 1 bar.
The long postequilibration is the longest of the depot examples,
intended to let the cured network relax meaningfully before the
final coordinates are written.  Postcure wall-clock: ~40 minutes.

Profile
^^^^^^^

End-of-run stage profile from a representative single-CPU + single-GPU
run:

.. code-block:: text

   Stage                                                   wall      subprocess
   ------------------------------------------------------------------------------
   setup                                                27.09 s          6.51 s
   initialization                                       ~5 s             ~3 s
   densification                                        ~50 min          ~50 min
   precure                                            1h58m58s         1h58m58s
   cure                                              10h19m51s              0 ms
     iter-1                                           10m36s            9m33s
     iter-2                                            8m58s            8m12s
     ...
     iter-14                                          2h17m33s         2h15m19s
     iter-15                                          3h49m45s         3h47m07s
     capping                                              5 ms             0 ms
   postcure                                            38m05s          38m04s
   final                                               24.36 s             0 s

Total: ~13.5 hours.  Of that, gmx-mdrun consumes ~95 % of the
subprocess time; antechamber/parmchk2/tleap account for the rest of
the setup wall.

The next page covers the :ref:`results <htpb_results>`.
