Post-build simulations and analyses
-----------------------------------

Liquid styrene is not polymerized, so post-build "production" measurements like glass-transition temperature or Young's modulus are not meaningful here.  What we *can* do — and what is useful as a sanity check — is to equilibrate the as-built liquid further at 300 K and then measure its density and fractional free volume.

The :ref:`pde_tutorial` walks through the full set of postsim and analyze techniques in detail; this section just shows the minimum useful run for an unpolymerized liquid.

The postsim configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``postsim.yaml`` in the project base directory:

.. code-block:: yaml

    - equilibrate:
        input_top: systems/final-results/final.top
        input_gro: systems/final-results/final.gro
        T: 300
        ps: 100

Run it:

.. code-block:: console

    $ htpolynet postsim -cfg postsim.yaml -ocfg 0-liquid-styrene.yaml -proj proj-0

This produces ``proj-0/postsim/equilibrate/`` with the usual Gromacs outputs plus an on-the-fly density-vs-time plot.

Density during equilibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Placeholder
   :class: caution

   **TODO:** insert the ``rho_v_ns.png`` plot from ``proj-0/postsim/equilibrate/`` and report the mean equilibrated density (kg/m³ or g/cc) here.  Note whether the system has reached a stable density (i.e., the trace is flat over the second half of the trajectory).

Free volume
^^^^^^^^^^^

Use ``htpolynet analyze`` with a one-line config to invoke ``gmx freevolume`` on the equilibration trajectory.  Create ``fv.yaml``:

.. code-block:: yaml

    - command: freevolume

Then:

.. code-block:: console

    $ htpolynet analyze -cfg fv.yaml -proj proj-0

The output lands in ``proj-0/analyze/freevolume/``.

.. admonition:: Placeholder
   :class: caution

   **TODO:** report the fractional free volume (and standard error) from ``proj-0/analyze/freevolume/ffv.dat`` here.  For unpolymerized small-molecule liquids the FFV is typically in the 0.20–0.30 range; values much outside that suggest the equilibration was too short.
