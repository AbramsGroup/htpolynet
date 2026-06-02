.. _htpb_reactions:

Reactions
---------

This example uses **three reaction stages**:

1. **param-stage reactions** assemble small parameterized sub-units
   (dimers, trimers, quads) from the four small constituents.  Each
   product is a separately parameterized molecule.
2. **build-stage reactions** stitch those sub-units into the long
   ``DHT`` and ``THT`` HTPB chains.  Build-stage products inherit
   bonded parameters from the param-stage templates — no new GAFF
   parameterization runs at this stage.
3. **cure-stage reactions** form urethane bonds between the assembled
   HTPB chain hydroxyls and IPDI's isocyanate carbons.

Param-stage: small assembled sub-units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Six param-stage reactions produce the building blocks for the longer
chain assembly:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Product
     - Reaction
     - Role
   * - ``A2``
     - ``TB.C1 + TB.C4``
     - A trans-2-butene dimer; the core repeat building block.
   * - ``AO``
     - ``TB.C1 + TBO.C4``
     - A TB-TBO dimer; carries one end-cap.
   * - ``OBT``
     - ``OB.C3 + TB.C4``
     - A 1-butene + trans-2-butene dimer; chain initiator.
   * - ``OB3``
     - ``OBT.C4 + TB.C4``
     - A 3-residue OB-TB-TB trimer.
   * - ``A3``
     - ``A2.C2 + TB.C4``
     - A 3-residue all-TB trimer.
   * - ``A4``
     - ``A3.C4 + TB.C4``
     - A 4-residue all-TB quad; the central node for the branched
       ``THT`` chain.

A representative entry, ``A2``:

.. code-block:: yaml

   - name: dimer1
     stage: param
     reactants: {1: TB, 2: TB}
     product: A2
     atoms:
       A: {reactant: 1, resid: 1, atom: C1, z: 1}
       B: {reactant: 2, resid: 1, atom: C4, z: 1}
     bonds:
       - atoms: [A, B]
         order: 1

``stage: param`` is what tells htpolynet to actually run GAFF
parameterization on the assembled product (``A2`` becomes a real
GAFF-parameterized ``.mol2``/``.itp``).  Build-stage reactions later
will reuse these parameters when extending these dimers into longer
chains.

Build-stage: chain assembly via ``procession``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The heart of the chain-build is one ``polymerization`` reaction with
a ``procession`` directive that fires it 15 times:

.. code-block:: yaml

   - name: polymerization
     stage: build
     reactants: {1: A2, 2: TB}
     product: A18
     procession:
       increment_resid: 1
       count: 15
     atoms:
       A: {reactant: 1, resid: 2, atom: C1, z: 1}
       B: {reactant: 2, resid: 1, atom: C4, z: 1}
     bonds:
       - atoms: [A, B]
         order: 1

The ``procession`` block says "run this same reaction 15 times,
incrementing the ``A.resid`` by 1 each time".  Each iteration adds one
more ``TB`` residue onto the growing chain.  Starting from the 2-
residue ``A2``, after 15 iterations the chain is an 18-residue
``A18``.  htpolynet generates 16 successive products
(``A18_I0`` through ``A18_I15``, with each ``IK`` being the state
after the k-th iteration) and parameterizes them; you'll see them
flying past in the diagnostic log at setup time.

Once ``A18`` is assembled, additional build-stage reactions cap and
combine:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Product
     - Reaction
     - Role
   * - ``O19``
     - ``A18.C1 + TBO.C4``
     - Cap one end of an A18 chain with a TBO terminator.
   * - ``DHT1``
     - ``OB3 + O19``
     - Add a second 18-residue segment + cap.
   * - ``DHT``
     - ``DHT1 + O19``
     - Add the third 18-residue segment + cap; final linear chain.
   * - ``THT1``, ``THT2``, ``THT``
     - successive ``A4 + O19`` additions
     - Build a branched chain off the central ``A4`` quad,
       eventually capping all three arms with ``TBO``.

By the time the assembly is done, ``DHT`` is a linear chain of ~57
residues (OB + 2×A18 + 2×TBO end-caps + glue residues) and ``THT`` is
a branched chain with three arms meeting at a central ``A4`` node and
each ending in ``TBO``.

This whole pipeline runs at setup time only; once the templates are
parameterized, the cured-system build inserts whole ``DHT`` /
``THT`` molecules into the simulation box.

Cure-stage: urethane formation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two cure reactions, one per IPD isocyanate carbon:

.. code-block:: yaml

   - name: urethane-1
     stage: cure
     reactants: {1: TBO, 2: IPD}
     product: UR1
     probability: 1.0
     atoms:
       A: {reactant: 1, resid: 1, atom: O1, z: 1}
       B: {reactant: 2, resid: 1, atom: C1, z: 1}
     bonds:
       - atoms: [A, B]
         order: 1

   - name: urethane-2
     stage: cure
     reactants: {1: TBO, 2: IPD}
     product: UR2
     probability: 1.0
     atoms:
       A: {reactant: 1, resid: 1, atom: O1, z: 1}
       B: {reactant: 2, resid: 1, atom: C2, z: 1}
     bonds:
       - atoms: [A, B]
         order: 1

These are essentially the same reaction with different IPD-carbon
targets, encoding the two non-equivalent isocyanates on IPDI: ``C1``
is the primary isocyanate (on the secondary ring carbon ``C3``) and
``C2`` is the secondary isocyanate (on the methylene substituent).
Both fire at ``probability: 1.0`` — no kinetic preference is encoded
between them in this example, which is a simplification.  (In real
IPDI, ``C1`` tends to be more reactive at room temperature; a more
detailed model could give it ``probability: 1.0`` and ``C2``
``probability: 0.3-0.5``.)

Both produce a urethane bond:

.. code-block:: text

   HTPB-O ... + ... C(H)=O-N(H)-IPDI
   ─────────────────────────
   HTPB-O-C(=O)-N(H)-IPDI + 2 sacrificial H atoms lost

Network topology after cure: each IPD's two isocyanates bond to two
HTPB chain ends (potentially on different chains), so IPD acts as a
2-functional crosslinker bridging two HTPB hydroxyls.  Linear ``DHT``
chains contribute 2 hydroxyls each (100 total); branched ``THT``
chains contribute 3 (150 total); total 250 hydroxyls.  125 IPDs × 2
isocyanate carbons = 250 reactive sites on the IPD side.  Exact 1:1
stoichiometry, so at full conversion the network is fully bridged.

Symmetry expansion
^^^^^^^^^^^^^^^^^^

Unlike the BPA examples (where ``[O1, O2]`` symmetry-equivalence
auto-expands one reaction into many), this example does **not** use
``symmetry_equivalent_atoms`` on any constituent — every reactive
atom on every constituent gets its own unique name and gets its own
explicit reaction.  Combined with the 6 param-stage + 8 build-stage +
2 cure-stage reactions and one stereocenter expansion (IPD's ``C3``),
htpolynet reports 35 molecule templates at startup.

The next page walks through the :ref:`full YAML <htpb_configuration>`.
