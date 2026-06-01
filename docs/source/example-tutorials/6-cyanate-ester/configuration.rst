.. _badcy_configuration:

The Configuration File
----------------------

The complete ``6-cyanate-ester.yaml`` from the depot:

.. literalinclude:: ../../../../src/htpolynet/resources/example_depot/6-cyanate-ester.yaml
   :language: yaml

Most blocks (``Title``, ``gromacs``, ``ambertools``, ``densification``,
``precure``, ``CURE``, ``postcure``) follow the same conventions as
the earlier tutorials; only ``constituents``, ``reactions``, and the
new ``postcure_repair`` block carry tutorial-specific content.

``constituents``
^^^^^^^^^^^^^^^^

Three entries: ``BPA`` (the bisphenol-A bridge, 360 copies),
``TAZ`` (1,3,5-triazine, 240 copies — the trifunctional crosslinker),
and ``CYN`` (hydrogen cyanide, no count, used only as a
parameterization template for ``-C#N`` caps).  Covered on the
:ref:`monomers page <badcy_monomers>`.

The 3:2 stoichiometry (360 BPA × 2 reactive O = 240 TAZ × 3 reactive
C = 720 reactive sites on each side) is exact, so at full conversion
the network has no unreacted reactive sites of either kind.

``reactions``
^^^^^^^^^^^^^

Two entries: ``etherify`` (cure-stage; the only reaction that
actually fires during the cure loop) and ``cap_with_cyanate``
(repair-stage; a template-only directive consumed by the postcure
repair driver).  Walked through on the :ref:`reactions page
<badcy_reactions>`.

``postcure_repair``
^^^^^^^^^^^^^^^^^^^

The block at the bottom of the YAML drives the new postcure repair
stage:

.. code-block:: yaml

   postcure_repair:
     - type: triazine_to_cyanate_cap
       crosslinker:
         residue: TAZ
         ring_carbon_atoms: [C1, C2, C3]
         ring_nitrogen_atoms: [N1, N2, N3]
         full_bond_count: 3
       bridge:
         residue: BPA
         reactive_oxygen_atoms: [O1, O2]
       cap_residue: CYN
       cap_template: BPA~O1-C1~CYN
       cap_search_radius: 0.6

The list-of-dicts shape lets multiple repair drivers run in sequence;
in this YAML there's only one entry.  See the :ref:`postcure-repair
user-guide page <postcure_repair>` for the architecture and the
per-field reference.

What expansion produces
^^^^^^^^^^^^^^^^^^^^^^^

Combining the constituent declarations with the two reactions:

* ``etherify`` (one user reaction) symmetry-expands into six
  linked-product templates (``BPA.{O1,O2} × TAZ.{C1,C2,C3}``,
  products ``BPA~O1-C1~TAZ`` through ``BPA~O2-C3~TAZ``);
* ``cap_with_cyanate`` (one user reaction) expands into two
  (``BPA.{O1,O2} × CYN.C1``, products ``BPA~O1-C1~CYN`` and
  ``BPA~O2-C1~CYN``).

So the user wrote 2 reactions and ``htpolynet`` produces 6 cure
templates + 2 repair templates, plus the 3 constituent monomers
themselves — 11 templates total.

Compared to a cyclotrimerization model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The full chemistry of the run is intentionally simple compared to a
literal cyclotrimerization model.  The trade-off:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - **Topological model (this tutorial)**
     - **Literal cyclotrimerization**
   * - Triazine ring exists in the monomer; cure is A2+B3 ether
       substitution.
     - Triazine ring is formed during cure from three R-O-C#N
       end-groups closing into a ring.
   * - No ring closure during cure; uses only existing
       ``etherify``-style A2+B3 cure machinery.
     - Requires 3-way C-N bond closure and ``bondcycle_collective``
       bypasses for the heteroatom ring.
   * - Incomplete cure → free BPA-OH + bare triazine C-H artefacts;
       postcure repair stage converts them to BPA-O-C#N residuals.
     - Incomplete cure → free BPA-O-C#N end-groups directly (no
       repair needed; chain ends are physically correct as-is).
   * - Final cured-network topology and atom inventory are
       structurally identical to a cyclotrimerized BADCy thermoset.
     - Same final network, but cure kinetics realism preserved.
   * - Cure-kinetics or intermediate-species questions cannot be
       addressed.
     - Cure-kinetics realism preserved; but the 3-way ring closure is
       awkward to drive deterministically.

If you want the network as a target without cure-kinetics realism,
this tutorial's approach is simpler and avoids the cyclotrimerization
machinery.  If you want cure kinetics, you'd need an open-CY-based
model along the lines of the historical example 6 implementation
(pre-2.1); the relevant building blocks (formaldimine ``CY`` etc.)
remain in the htpolynet codebase.

Next is :ref:`actually running the build <badcy_run>`.
