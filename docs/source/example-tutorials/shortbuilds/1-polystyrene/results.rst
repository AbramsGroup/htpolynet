.. _ps_results:

Results
-------

After the build completes, the project directory ``proj-0/`` contains the final Gromacs topology (``*.top``) and coordinate (``*.gro``) files for the cured polystyrene system, along with a full archive of intermediate files.

Using ``htpolynet plots``
^^^^^^^^^^^^^^^^^^^^^^^^^

``htpolynet`` ships with a plotting subcommand that reads ``diagnostics.log`` and generates summary figures:

.. code-block:: console

    $ htpolynet plots -diag diagnostics.log

This produces plots of conversion vs. time, density evolution, and per-iteration bond counts, which give a quick picture of how the cure progressed.

As a further exercise, copy ``pSTY.yaml`` to ``pSTY-low.yaml``, change ``desired_conversion`` from ``0.95`` to ``0.50``, and run a second build:

.. code-block:: console

    $ cp pSTY.yaml pSTY-low.yaml
    $ # edit pSTY-low.yaml: desired_conversion: 0.50
    $ htpolynet run -diag diagnostics-low.log pSTY-low.yaml &> console-low.log &

The second build will populate ``proj-1/``.  Comparing plots from ``diagnostics.log`` and ``diagnostics-low.log`` shows how cure conversion affects system density and chain structure.

Post-simulation analyses
^^^^^^^^^^^^^^^^^^^^^^^^

Once you have a cured polystyrene system, you can use ``htpolynet postsim`` and ``htpolynet analyze`` to run production MD and compute properties such as the glass-transition temperature.  These workflows are described in the :ref:`post-simulation analyses tutorial <tutorials_postsim_analyses>`.
