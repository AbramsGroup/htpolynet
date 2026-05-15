.. _ps_configuration:

The Configuration File
----------------------

The complete ``1-polystyrene.yaml`` carried by the depot:

.. literalinclude:: ../../../../../src/htpolynet/resources/example_depot/1-polystyrene.yaml
   :language: yaml

A block-by-block walk-through (skipping the blocks already covered in
:ref:`example 0 <liquid_styrene_configuration>` — ``Title``,
``gromacs``, ``ambertools``, ``densification``, ``precure``):

``constituents``
   One species: ``STY``, 1000 copies, generated from SMILES.  See
   :ref:`monomer <ps_monomer>` for the atom-mapping details.

``CURE``
   Controls the Connect-Update-Relax-Equilibrate cycle:

   * ``controls.search_radius`` (0.5 nm) and
     ``controls.radial_increment`` (0.25 nm) set the starting search
     radius for finding reactive pairs and how aggressively to grow it
     when nothing is found.
   * ``controls.max_iterations`` caps the number of CURE iterations
     even if conversion has not yet been reached.
   * ``controls.desired_conversion`` (0.95) is the target.  Conversion
     is defined relative to the maximum possible bonds for the system.
   * ``controls.late_threshold`` (0.85) is the conversion above which
     ``htpolynet`` stops applying the per-reaction ``probability``
     filter — once you are close to saturation, every eligible bond is
     worth taking.
   * ``drag`` configures the optional preupdate dragging step that
     pulls *almost*-eligible pairs together before bond formation.
     With ``trigger_distance: 0.6 nm``, ``increment: 0.08 nm``, and
     ``limit: 0.3 nm``, ``htpolynet`` will progressively shorten
     candidate bond distances by adding a restraint and running short
     MD between increments.
   * ``relax`` configures the bond-relaxation step that runs after
     every CURE topology update — minimization plus a 1 ps NVT at 600 K
     plus a 2 ps NPT at 600 K / 1 bar.
   * ``equilibrate`` runs a 100 ps NPT at 300 K / 1 bar at the end of
     each iteration.
   * ``gromacs.rdefault`` (0.9 nm) is the nonbonded cutoff used for
     CURE-stage MD runs.

   .. tip::

      The ``CURE.controls.min_bonds_per_iteration`` knob (default
      ``10``) sets the minimum number of bonds an iteration must find
      before it stops growing the search radius.  Bigger values batch
      more bonds per iteration and so run fewer relax/equilibrate
      cycles overall, at the cost of more strained bonds in each
      batch.  ``htpolynet`` automatically clamps the effective floor
      against how many bonds are still possible, so demanding
      ``min_bonds_per_iteration: 50`` near end-of-cure won't stall the
      build.  Empirically, ``10`` is a good operating point for the
      epoxy/amine examples in this tutorial set: see :ref:`pde_run`
      for the measured iteration counts on DGEBA/PACM.

``postcure``
   The post-cure equilibration cascade: two more 300/600 K annealing
   cycles followed by a 100 ps NPT at 300 K / 1 bar.

``reactions``
   The cure and cap reactions described on the previous page.
