.. _ps_run:

Running the Build
=================

From inside ``1-polystyrene/``, launch ``htpolynet run``:

.. code-block:: console

    $ htpolynet run -diag diagnostics.log pSTY.yaml &> console.log &

The ``-diag`` flag writes per-iteration diagnostic data to ``diagnostics.log`` (used later for plotting).  Redirecting stdout/stderr to ``console.log`` lets you monitor progress with ``tail -f console.log``.

Build stages
^^^^^^^^^^^^

``htpolynet`` works through the following stages in order.  Each is visible in ``console.log``.

1. **Template parameterization** — ``htpolynet`` parameterizes the monomer ``STY`` and automatically generates and parameterizes all chain-expanded dimer, trimer, and tetramer templates using AmberTools (``antechamber``, ``parmchk2``, ``tleap``) and Gromacs (energy minimization).

2. **System assembly** — 200 STY molecules are placed in a cubic box at the requested initial density of 300 kg/m\ :sup:`3`.

3. **Densification** — a short minimization followed by NVT (10 ps, 300 K) and NPT (200 ps, 300 K, 10 bar) MD simulations compress the box to near the ambient density.

4. **Pre-cure equilibration** — an NPT MD run at 1 bar and 300 K is followed by two annealing cycles (300 → 600 → 300 K) and a final 100 ps NPT equilibration.

5. **CURE iterations** — the CURE algorithm searches for reactive C1–C2 pairs within the current search radius, optionally drags distant pairs closer, deletes sacrificial H atoms, forms new bonds, relaxes bond lengths, and runs a short NPT equilibration.  The radius expands by 0.25 nm whenever no new bonds are found.  This repeats until 95% conversion is reached or 150 iterations are exhausted.

6. **Cap reactions** — unreacted STY monomers have their C1–C2 bond restored to a double bond.

7. **Post-cure equilibration** — two more annealing cycles followed by a final NPT equilibration produce the finished ``proj-0/`` output.

Monitoring progress
^^^^^^^^^^^^^^^^^^^

To watch the build in real time:

.. code-block:: console

    $ tail -f console.log

To check how many CURE iterations have completed:

.. code-block:: console

    $ grep "^INFO> Iteration" console.log

When the build finishes successfully, the last lines of ``console.log`` will indicate that post-cure equilibration has completed and the final ``top`` and ``gro`` files have been saved to ``proj-0/``.
