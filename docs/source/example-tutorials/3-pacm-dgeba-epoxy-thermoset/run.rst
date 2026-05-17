.. _pde_run:

Running the Build
-----------------

From inside the working directory containing
``3-pacm-dgeba-epoxy-thermoset.yaml``:

.. code-block:: console

   $ htpolynet run -diag diagnostics.log 3-pacm-dgeba-epoxy-thermoset.yaml &> console.log &

Stage layout under ``proj-N/systems/`` is the same as
:ref:`example 1 <ps_run>` — ``init/``, ``densification/``, ``precure/``,
``iter-K/``, ``capping/``, ``postcure/``, ``final-results/`` — plus
``plots/`` and ``profile.json`` at the project root.

Setup
^^^^^

At ``setup`` time ``htpolynet`` parameterizes all 22 molecule templates
discussed in the :ref:`configuration page <pde_configuration>`.  An
abbreviated view of the diagnostic log:

.. code-block:: text

   INFO> Templates in proj-0/molecules/parameterized
   INFO> 22 molecules detected in 3-pacm-dgeba-epoxy-thermoset.yaml
   INFO>                       explicit: 5
   INFO>     implied by stereochemistry: 6
   INFO>            implied by symmetry: 11
   INFO> AmberTools> generating GAFF parameters from PAC.mol2
   INFO> PAC: 210.36 g/mol
   INFO> AmberTools> generating GAFF parameters from DGE.mol2
   INFO> DGE: 344.43 g/mol
   INFO> AmberTools> generating GAFF parameters from PAC~N1-C1~DGE.mol2
   INFO> PAC~N1-C1~DGE: 552.78 g/mol
   INFO> AmberTools> generating GAFF parameters from DGEC.mol2
   INFO> DGEC: 342.42 g/mol
   INFO> AmberTools> generating GAFF parameters from PAC~N1-C1~DGE-C1~DGE.mol2
   INFO> PAC~N1-C1~DGE-C1~DGE: 895.19 g/mol
   ...
   INFO> Generated 22 molecule templates
   INFO> Initial composition is DGE 200, PAC 100
   INFO> 100% conversion is 400 bonds

Densification + precure
^^^^^^^^^^^^^^^^^^^^^^^

This is a more compact monomer pair than the bisGMA/styrene system,
so the starting density (300 kg/m³) is higher and a single 300 ps NPT
suffices to reach ~1.05 g/cm³ — no ``repeat: 8`` trick needed.

.. code-block:: text

   INFO> Densification in proj-0/systems/densification
   INFO> Running Gromacs: minimization
   INFO> Running Gromacs: nvt ensemble;  10.00 ps,  300.00 K
   INFO> Running Gromacs: npt ensemble; 300.00 ps,  300.00 K,  10.00 bar
   INFO> Current box side lengths: 5.225 nm x 5.225 nm x 5.225 nm
   INFO> Density                      1046.77

CURE iterations
^^^^^^^^^^^^^^^

Target conversion is 0.95, so the cure aims for 380 of the maximum
400 bonds.  Early iterations are productive (dozens of bonds at a
time); the long tail picks up bonds in ones and twos with the
search-radius growth machinery doing the work.  A typical run takes
~25–35 iterations.  Excerpted:

.. code-block:: text

   INFO> ~~~~~~~~~~~~~~ Iteration 1 begins ~~~~~~~~~~~~~~
   INFO> Bond search using radius 0.5 nm initiated
   INFO> Iteration 1 will generate 71 new bonds
   ...
   INFO> Iteration 1 current conversion 0.177 or 71 bonds
   INFO> ~~~~~~~~~~~~~~ Iteration 2 begins ~~~~~~~~~~~~~~
   INFO> Bond search using radius 0.5 nm initiated
   INFO> Iteration 2 will generate 50 new bonds
   ...
   INFO> Iteration 2 current conversion 0.302 or 121 bonds
   ...
   INFO> ~~~~~~~~~~~~~~ Iteration 32 begins ~~~~~~~~~~~~~~
   INFO> Bond search using radius 0.5 nm initiated
   INFO> Iteration 32 will generate 2 new bonds
   INFO> Step "cure_drag" initiated on 2 distances (max 0.720 nm)
   ...
   INFO> Iteration 32 current conversion 0.950 or 380 bonds

Notice the late iterations engaging the ``cure_drag`` step — when the
remaining reactive pairs are far apart, ``htpolynet`` adds a harmonic
restraint and pulls them together over short MD before forming the
bond and relaxing.  This is also where the
``CURE.controls.min_bonds_per_iteration`` knob has its biggest effect:
raising it batches the small late-stage finds together and collapses
many ``iter-K`` directories into fewer, larger ones.

Measured iteration counts on this system:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``min_bonds_per_iteration``
     - Iterations to reach 95% cure
   * - ``1``
     - 41
   * - ``10`` (default)
     - 15
   * - ``20``
     - 14

So the default of ``10`` is roughly a 2.7× iteration-count cut over
the un-batched (``1``) regime, and the marginal benefit from raising
further to ``20`` is small — diminishing returns set in fast once you
clear the floor of "every late-stage iteration finds one or two bonds
at a time".

Capping and postcure
^^^^^^^^^^^^^^^^^^^^

20 oxirane rings are unreacted at 95 % conversion, so capping has real
work to do:

.. code-block:: text

   INFO> Capping begins
   INFO> Capping will generate 20 new bonds
   INFO> Step "cap_relax" initiated on 20 distances (max 0.252 nm)
   ...
   INFO> Connect-Update-Relax-Equilibrate (CURE) ends

The final density after postcure landed around 1.09 g/cm³ in a
representative run.  ``htpolynet`` then writes ``final.gro``,
``final.top``, ``final.tpx``, ``final.grx``, the VMD viz pair, and the
profile report.  The next page covers a few of the diagnostic plots:
:ref:`pde_results`.
