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

The 20 densification NPT repeats at 600 K / 10 bar progressively
compact a dilute initial state into a near-melt density of
~0.9-1.0 g/cm³.  Each repeat is 100 ps; the full densification
takes ~40 minutes of wall clock.  Precure adds a 300 ps NPT
preequilibration at 300 K / 1 bar, then an anneal cycle (two
cycles between 300 and 600 K, 200 ps per segment) so the chains can
explore conformational space before cure starts.  Total precure
wall-clock: ~1 hour.

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
     - 42
     - 0.168
     - 10:56
   * - 2
     - 31
     - 0.292
     - 10:43
   * - 3
     - 21
     - 0.376
     - 10:40
   * - 4
     - 23
     - 0.468
     - 10:37
   * - 5
     - 13
     - 0.520
     - 10:16
   * - 6
     - 10
     - 0.560
     - 10:26
   * - 7
     - 14
     - 0.616
     - 16:20
   * - 8
     - 18
     - 0.688
     - 17:28
   * - 9
     - 14
     - 0.744
     - 17:28
   * - 10
     - 11
     - 0.788
     - 22:31
   * - 11
     - 10
     - 0.828
     - 22:23
   * - 12
     - 11
     - 0.872
     - 47:35
   * - 13
     - 10
     - 0.912
     - 2:15:17
   * - 14
     - 8
     - 0.944
     - 3:51:37
   * - 15
     - 1
     - 0.948
     - 1:06:10

The cure-tail effect is sharp: the first 11 iterations together
take ~3 hours; the last 4 take another ~7.  By the late iterations
only a handful of hydroxyl / isocyanate pairs are left unbonded,
finding pairs within the bond-search radius requires the
``cure_drag`` step to pull distant atoms together over multiple MD
segments, and each ``cure_drag`` cascade scales with the
inter-atom separation.  ``min_bonds_per_iteration: 10`` is what
keeps the iteration count from blowing up to 50+ at the tail;
raising it further would slightly reduce iteration count but each
iteration would have to drag further-apart atoms together, with
diminishing returns.

Total cure wall-time: ~10.7 hours.  Capping is trivially fast (0
bonds — all reactive sites that were going to bond did) and runs in
milliseconds.

Postcure
^^^^^^^^

Postcure runs two anneal cycles between 300 K and 600 K (50 ps per
segment) followed by a 200 ps NPT postequilibration at 300 K /
1 bar to let the cured network relax meaningfully before the final
coordinates are written.  Postcure wall-clock: ~17 minutes.

Profile
^^^^^^^

End-of-run stage profile from a representative single-CPU + single-GPU
run:

.. code-block:: text

   Stage                                                   wall      subprocess
   ------------------------------------------------------------------------------
   setup                                                28.11 s          7.77 s
   initialization                                       ~5 s             ~3 s
   densification                                        ~40 min          ~40 min
   precure                                              57m26s          57m26s
   cure                                              10h40m27s              0 ms
     iter-1                                           10m56s            9m52s
     iter-2                                           10m43s            9m46s
     ...
     iter-14                                          3h51m37s         3h48m04s
     iter-15                                          1h06m10s         1h04m22s
     capping                                              5 ms             0 ms
   postcure                                            16m37s          16m37s
   final                                               24.34 s             0 s

Total: ~12 hours.  Of that, gmx-mdrun consumes ~95 % of the
subprocess time; antechamber/parmchk2/tleap account for the rest of
the setup wall.

The next page covers the :ref:`results <htpb_results>`.
