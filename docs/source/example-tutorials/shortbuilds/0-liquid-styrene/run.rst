.. _liquid_styrene_run:

Running the Build
-----------------

From inside the working directory containing ``0-liquid-styrene.yaml``:

.. code-block:: console

   $ htpolynet run -diag diagnostics.log 0-liquid-styrene.yaml &> console.log &

This kicks off the full workflow:

1. Parameterize the styrene monomer (``htpolynet`` invokes
   ``antechamber`` / ``tleap`` / ``parmchk2`` under the hood, after
   first generating ``STY.mol2`` from the SMILES in the config).
2. Pack 200 copies of the monomer into a low-density initial box.
3. Run the densification cascade defined under ``densification`` and
   then the preequilibration + anneal cascade under ``precure``.

Each stage's output lives in a separate subdirectory under
``proj-N/systems/`` (where ``N`` is the next available index — see the
``Working in new project`` line at the top of ``diagnostics.log``).

.. todo::

   - List the stage subdirectories produced (``init/``,
     ``densification/``, ``precure/``, ``final-results/``) and what's in
     each.
   - Note the absence of ``postcure`` / ``capping`` / ``iter-*`` dirs in
     this example (those only appear when ``CURE`` is configured).
   - Show what to look for in ``diagnostics.log`` to confirm the run
     completed cleanly.
