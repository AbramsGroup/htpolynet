.. _ps_introduction:

Introduction
------------

The first step is to get to a clean working directory, and then use ``htpolynet fetch-example`` to set up this example.

.. code-block:: console

   $ mkdir my_polystyrene
   $ cd my_polystyrene
   $ htpolynet fetch-example 1
   $ ls
   1-polystyrene.sh

``fetch-example`` copies the shell script ``1-polystyrene.sh`` into the current directory.  Running it creates the ``1-polystyrene/`` project directory tree, generates the monomer ``mol2`` file, and writes the ``htpolynet`` configuration file:

.. code-block:: console

   $ bash 1-polystyrene.sh
   $ ls 1-polystyrene
   lib/  pSTY.yaml
   $ tree 1-polystyrene
   1-polystyrene/
   ├── lib
   │   └── molecules
   │       ├── inputs
   │       │   ├── STY.mol2
   │       │   └── STY.png
   │       └── parameterized

The ``1-polystyrene/lib/molecules/inputs/`` directory now contains the monomer structure file ``STY.mol2``.
The build is launched from inside ``1-polystyrene/``:

.. code-block:: console

   $ cd 1-polystyrene
   $ htpolynet run -diag diagnostics.log pSTY.yaml &> console.log &
