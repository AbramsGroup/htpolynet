.. _badcy_run:

Running the Build
-----------------

From inside the working directory containing
``6-cyanate-ester.yaml``:

.. code-block:: console

   $ htpolynet run -diag diagnostics.log 6-cyanate-ester.yaml &> console.log &

The stage layout under ``proj-N/systems/`` matches earlier examples
(``init/``, ``densification/``, ``precure/``, ``iter-K/``,
``capping/``, ``postcure/``, ``final-results/``, plus ``plots/`` and
``profile.json`` at the project root) **and adds a new** ``repair/``
**directory** between ``capping/`` and ``postcure/``.  The repair
stage writes its ``repaired.gro``/``repaired.top``/``repaired.tpx``
plus a steepest-descent + short NVT relaxation pair there so the
modified topology has a chance to settle before the postcure MD
ensemble takes over.

Setup
^^^^^

``htpolynet`` parameterizes the 11 templates discussed in the
:ref:`configuration page <badcy_configuration>`:

.. code-block:: text

   INFO> 11 molecules detected in 6-cyanate-ester.yaml
   INFO>                       explicit: 5
   INFO>     implied by stereochemistry: 0
   INFO>            implied by symmetry: 6
   INFO> AmberTools> generating GAFF parameters from BPA.mol2
   INFO> BPA: 228.28 g/mol
   INFO> AmberTools> generating GAFF parameters from TAZ.mol2
   INFO> TAZ: 81.08 g/mol
   INFO> AmberTools> generating GAFF parameters from CYN.mol2
   INFO> CYN: 27.03 g/mol
   INFO> AmberTools> generating GAFF parameters from BPA~O1-C1~TAZ.mol2
   INFO> BPA~O1-C1~TAZ: 307.35 g/mol
   INFO> AmberTools> generating GAFF parameters from BPA~O1-C1~CYN.mol2
   INFO> BPA~O1-C1~CYN: 253.29 g/mol
   ...
   INFO> Generated 11 molecule templates
   INFO> Initial composition is BPA 360, TAZ 240
   INFO> 100% conversion is 720 bonds

The molecular weights are a useful quick sanity check: 228.28 (BPA) +
81.08 (TAZ) – 2 × 1.008 (lost H atoms) = 307.35 (``BPA~O1-C1~TAZ``);
228.28 (BPA) + 27.03 (HCN) – 2 × 1.008 = 253.29
(``BPA~O1-C1~CYN``).

Densification + precure
^^^^^^^^^^^^^^^^^^^^^^^

200 kg/m³ initial density and a 100 ps NPT segment (× 4 repeats) bring
the box to roughly 1.0 g/cm³ before precure.  Precure runs the
preequilibration (200 ps NPT at 300 K, 1 bar) and a short anneal cycle
between 300 and 500 K to relax any high-energy contacts from the
random initial placement.

Cure
^^^^

CURE runs until either ``desired_conversion: 0.90`` or
``max_iterations: 150`` is reached.  On a representative run the cure
converges in nine iterations:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Iteration
     - Bonds formed
     - Cumulative conversion
     - Wall time
   * - 1
     - 160
     - 0.222
     - 2:21
   * - 2
     - 153
     - 0.435
     - 2:17
   * - 3
     - 137
     - 0.625
     - 2:21
   * - 4
     - 88
     - 0.747
     - 2:08
   * - 5
     - 52
     - 0.819
     - 1:58
   * - 6
     - 31
     - 0.863
     - 1:50
   * - 7
     - 14
     - 0.882
     - 1:45
   * - 8
     - 11
     - 0.897
     - 1:42
   * - 9
     - 2
     - 0.900
     - 1:25

The classic long tail: 80 % cure in 4 iterations, the remaining 10 %
takes another 5.  No capping work because ``etherify`` is the only
cure reaction and the cap stage has nothing to do (no cap directives
in the YAML).

Repair
^^^^^^

After cure converges, the postcure topology-repair stage fires:

.. code-block:: text

   INFO> ************ Postcure repair in proj-0/systems/repair *************
   INFO> triazine_to_cyanate_cap: 63 incomplete TAZ residues identified (189 caps total, 72 free fragments to donate)
   INFO> triazine_to_cyanate_cap: redistributing residual charge -23.3716 across 144 repaired-residue neighbours
   INFO> ******** Postcure repair performed 63 dismantle operations ********
   INFO> Relaxing repaired geometry
   INFO> Running Gromacs: minimization
   INFO> Running Gromacs: nvt ensemble;   5.00 ps,  300.00 K

Decoding the numbers:

* **63 incomplete TAZ residues** out of 240 total — i.e. 177 of the
  240 triazines (~74 %) reached the full 3-bonded state during cure.
  Each incomplete one carries between 0 and 2 bonded BPAs.
* **189 caps total** = 63 × 3.  Each dismantled ring is split into
  three independent -C#N fragments.
* **72 free fragments to donate** = the number of dangling triazine
  C atoms across all incomplete rings.  This is also the number of
  unreacted BPA-OH groups (by atom conservation), so the matching is
  exact and every free fragment finds a home.
* **189 - 72 = 117 in-place caps**: fragments whose ring C atom was
  already bonded to a BPA during cure, so the BPA-O-C bond is
  preserved and only the atom types, bond orders, and angle/dihedral
  parameters update from the templated BPA-O-C#N values.
* **Residual charge ≈ -23 e** distributed across **144 atoms** =
  72 × 2 (the BPA-O atoms newly bonded to free caps, plus the
  CYN-C atoms whose H was deleted).  This is the charge from the
  deleted sacrificial H atoms, redistributed across the heavy-atom
  neighbours so the system stays net-neutral for Ewald.

The repair stage finishes by running a steepest-descent minimization
and a short (5 ps) NVT settle on the modified topology, so any LJ
clashes introduced by physically relocating the free-cap atoms get
relaxed before postcure MD starts.

Postcure
^^^^^^^^

Postcure runs the standard anneal (between 300 K and 500 K, two
cycles) followed by a 100 ps NPT postequilibration at 300 K and 1 bar.
The final density typically lands around 1.1 g/cm³ — a touch lower
than fully cured BADCy (≈ 1.2 g/cm³) because of the residual ``-C#N``
end-groups breaking the network into smaller clusters.

Profile
^^^^^^^

End-of-run stage profile (representative run, 4-core CPU + 1 GPU):

.. code-block:: text

   Stage                                                   wall      subprocess
   ------------------------------------------------------------------------------
   setup                                                 721 ms            0 ms
   initialization                                        ~10 s            ~5 s
   densification                                         ~3 min          ~3 min
   precure                                              3m28s           3m27s
   cure                                                17m46s              0 ms
     iter-1                                             2m22s          1m24s
     iter-2                                             2m17s          1m22s
     iter-3                                             2m21s          1m30s
     ...
   repair                                               1m09s           12 s
   postcure                                             1m55s          1m55s
   final                                                5 s              0 s

The cure dominates the run as expected.  The ``repair`` stage's wall
time (~1 min) is split between the surgery itself (~5 s — the
remaining time the minimization and 5-ps NVT settle).  The
``proj-0/profile.json`` file carries the same data in
machine-readable form.

Next is the :ref:`results page <badcy_results>`.
