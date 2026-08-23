.. _building_a_system:

Building a System, Start to Finish
----------------------------------

The other pages in this guide are reference: what each subcommand accepts,
what each configuration directive means, what the program does internally.
This page is the procedure — the order to do things in, and where each step
can go wrong.  It assumes ``htpolynet`` is installed (see :doc:`/install`)
and that ``htpolynet info`` reports AmberTools and Gromacs on your ``PATH``.

Start from the nearest example, not from an empty file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A working configuration is a few hundred lines and most of it is not
chemistry-specific.  Writing one from scratch is the slowest possible route.
Pick whichever bundled example is closest in *reaction topology* to what you
want and edit it:

.. code-block:: console

    $ htpolynet fetch-example 1     # linear chain growth (polystyrene)
    $ htpolynet fetch-example 2     # A2 + B4 thermoset (bisGMA/styrene)
    $ htpolynet fetch-example 3     # amine + epoxy (PACM/DGEBA)
    $ htpolynet fetch-example 6     # cyclotrimerization (cyanate ester)

Closeness in topology matters more than closeness in chemistry.  If you are
building a step-growth thermoset from a difunctional and a tetrafunctional
monomer, example 2 is a better starting point than an example that happens
to share a functional group but polymerizes by chain growth.

Run the example unmodified once before editing it.  It confirms your
toolchain works, and it gives you a known-good log to compare against when
your own configuration misbehaves.

Describe monomers in their *active* form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the single most common conceptual error, and it produces a build
that runs to completion and is wrong rather than one that fails.

``htpolynet`` conserves valence: when two atoms bond, each gives up a
**sacrificial hydrogen**.  So the structure you describe is not the textbook
monomer — it is the monomer with its reactive sites already saturated.
Styrene is described as ethylbenzene.  See
:ref:`molecular_structure_inputs` for the full explanation and pictures;
do not skip it.

Prefer writing SMILES directly into the ``constituents`` block with
RDKit atom-map labels, which names reactive atoms by chemical identity
rather than by file index:

.. code-block:: yaml

    constituents:
      STY:
        smiles: "c1ccccc1[CH2:1][CH3:2]"
        reactive_atoms: {1: C1, 2: C2}

Watch the hydrogen counts inside brackets.  ``[C:1]`` means *zero* implicit
hydrogens; for an sp³ carbon you nearly always want ``[CH2:1]`` or
``[CH3:1]``.  A wrong count typically surfaces much later as a missing GAFF
angle parameter in ``tleap``, which is a long way from its cause.

Check what you can before spending compute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ htpolynet input-check my-system.yaml

This reports the atom count of the initial system, which is what you size a
core request against.  It is fast and worth running every time you change
monomer counts.

Be aware of what it does **not** yet check.  It does not verify that
``symmetry_equivalent_atoms`` groups really are topologically equivalent,
that your A2 + B3 site counts balance, or that ``desired_conversion`` is
reachable given the monomer counts you supplied.  Those are all silent
failure modes today: a wrong symmetry group generates reaction templates
the cure stage will simply never match, and the build spends its full
wall-clock reaching a conversion it could never have reached.  If a cure
stalls well below the conversion you asked for, suspect these first.

Run it
^^^^^^

.. code-block:: console

    $ htpolynet run my-system.yaml

Locally this is fine for the bundled examples.  For anything production-sized,
generate a batch script rather than writing one:

.. code-block:: console

    $ htpolynet gen-slurm-script my-system.yaml

Size the core request from the ``input-check`` atom count.  If you are using
the container image, target **CPU partitions**: its Gromacs comes from
conda-forge and is built against OpenCL, which Gromacs no longer uses to
drive NVIDIA devices, so the image cannot use a GPU.  Requesting
``--gres=gpu`` or passing ``--nv`` buys nothing and may cost you queue time.
See :doc:`container-usage`.

Parameterization runs first and is cached across projects and across
directories.  The first run of a new chemistry pays several minutes per
species; later runs reuse that work.  Read
:ref:`parameterization_caching` before you rely on it — in particular,
editing a monomer's structure without renaming it will silently reuse the
old parameterization.

Replicas, and what "the same" means
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``htpolynet run`` in the same base directory creates its own project
directory, so replicas are just repeated invocations of the same
configuration.

There is currently **no seed control**.  Conformer selection, the per-bond
probability test, and Gromacs velocity generation are all unseeded, so two
runs of one configuration diverge.  This is convenient in one direction —
independent replicas of a quenched-disorder ensemble come free — but it
means a specific build cannot be reproduced exactly, and a failure seen
once may not reappear.  Record the commit you built with if the result
matters.

Look at what you built
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ htpolynet plots build --proj proj-0
    $ htpolynet make-viz -top final.top -gro final.gro

``plots build`` gives traces, the reaction graph, and cluster-size
distributions; ``make-viz`` regenerates VMD visualization files from an
existing topology/coordinate pair.  For thermomechanical properties, run
post-build simulations and analyses:

.. code-block:: console

    $ htpolynet postsim postsim.yaml --proj proj-0
    $ htpolynet plots post --proj proj-0
    $ htpolynet analyze analyze.yaml --proj proj-0

See :doc:`configs/configs-for-postsim` and :doc:`configs/configs-for-analyze`.

When something goes wrong
^^^^^^^^^^^^^^^^^^^^^^^^^

A short list of traps that are known rather than mysterious:

* **The cure stalls below your target conversion.**  Usually a reaction
  template that never matches: check ``symmetry_equivalent_atoms`` and your
  reactive-atom names before suspecting the chemistry.

* **A ``tleap`` error about a missing parameter.**  Usually an atom typed
  wrongly because of a hydrogen-count error in a SMILES bracket atom.

* **Results that do not match the configuration you are reading.**  Suspect
  a stale cache entry: a structure edit without a rename reuses the old
  parameterization.  ``--force-parameterization`` rebuilds.

* **A build that dies during cure.**  ``-restart`` is documented as
  experimental and is currently broken at the cure stage, which is the worst
  place to lose work since cure is the longest phase.  Treat a dead cure as
  a restart from the beginning.

* **Anything else.**  Run ``htpolynet info`` and include its output when
  reporting a problem; it reports the versions of every external tool the
  build depends on.
