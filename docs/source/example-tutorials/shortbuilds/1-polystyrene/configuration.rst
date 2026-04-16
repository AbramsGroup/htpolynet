.. _ps_configuration:

The Configuration File
======================

The complete ``pSTY.yaml`` file written by ``1-polystyrene.sh`` is shown below.  A detailed explanation of every directive appears in the :ref:`configuration-file reference <configuration_run_example>`.

.. code-block:: yaml

  Title: polystyrene
  gromacs:
    gmx: gmx
    gmx_options: -quiet -nobackup
    mdrun: gmx mdrun
    mdrun_options:
      gpu_id: 0
  ambertools:
    charge_method: gas
  constituents:
    STY:
      count: 200
  densification:
    initial_density: 300.0  # kg/m3
    equilibration:
      - ensemble: min
      - ensemble: nvt
        temperature: 300
        ps: 10
      - ensemble: npt
        temperature: 300
        pressure: 10
        ps: 200
  precure:
    preequilibration:
      ensemble: npt
      temperature: 300        # K
      pressure: 1             # bar
      ps: 200
    anneal:
      ncycles: 2
      initial_temperature: 300
      cycle_segments:
        - T: 300
          ps: 0
        - T: 600
          ps: 20
        - T: 600
          ps: 20
        - T: 300
          ps: 20
        - T: 300
          ps: 20
    postequilibration:
      ensemble: npt
      temperature: 300        # K
      pressure: 1             # bar
      ps: 100
  CURE:
    controls:
      initial_search_radius: 0.5  # nm
      radial_increment: 0.25      # nm
      max_iterations: 150
      desired_conversion: 0.95
      late_threshhold: 0.85
    drag:
      trigger_distance: 0.6   # nm
      increment: 0.08         # nm
      limit: 0.3              # nm
      equilibration:
        - ensemble: min
        - ensemble: nvt
          temperature: 600
          nsteps: 1000
        - ensemble: npt
          temperature: 600
          pressure: 1
          nsteps: 2000
    relax:
      increment: 0.08         # nm
      equilibration:
        - ensemble: min
        - ensemble: nvt
          temperature: 600
          nsteps: 1000
        - ensemble: npt
          temperature: 600
          pressure: 1
          nsteps: 2000
    equilibrate:
      ensemble: npt
      temperature: 300       # K
      pressure: 1            # bar
      ps: 100
    gromacs:
      rdefault: 0.9          # nm
  postcure:
    anneal:
      ncycles: 2
      initial_temperature: 300
      cycle_segments:
        - T: 300
          ps: 0
        - T: 600
          ps: 20
        - T: 600
          ps: 20
        - T: 300
          ps: 20
        - T: 300
          ps: 20
    postequilibration:
      ensemble: npt
      temperature: 300       # K
      pressure: 1            # bar
      ps: 100
  reactions:
    - name:        sty1_1
      stage:       cure
      reactants:
        1: STY
        2: STY
      product:     STY~C1-C2~STY
      probability: 1.0
      atoms:
        A:
          reactant: 1
          resid: 1
          atom: C1
          z: 1
        B:
          reactant: 2
          resid: 1
          atom: C2
          z: 1
      bonds:
        - atoms: [A, B]
          order: 1
    - name:         styCC
      stage:        cap
      reactants:
        1: STY
      product:      STYCC
      probability:  1.0
      atoms:
        A:
          reactant: 1
          resid: 1
          atom: C1
          z: 1
        B:
          reactant: 1
          resid: 1
          atom: C2
          z: 1
      bonds:
        - atoms: [A, B]
          order: 2

A few things to note:

* ``charge_method: gas`` uses gas-phase AM1-BCC charges, which are faster than the default BCC charges and adequate for a tutorial run.
* The ``constituents`` block requests 200 styrene monomers — large enough to be interesting but small enough to finish in a few hours on a workstation.
* The ``CURE.gromacs.rdefault`` key sets the non-bonded cutoff radius for CURE-stage MD runs.
* All ``IMPORTANT DISCLAIMER`` caveats from the :ref:`tutorial index <example_tutorials_builds>` apply: this is *not* a production build.
