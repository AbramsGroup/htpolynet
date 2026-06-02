##############################
Installation and Prerequisites
##############################

A pre-built container image is available that bundles all required
software (Gromacs, AmberTools, OpenBabel, RDKit, and ``htpolynet``).  If
you have Docker or Singularity/Apptainer available, this is the easiest
way to get started — see :ref:`container_usage` for instructions.  The
remainder of this page describes how to install ``htpolynet`` and its
prerequisites directly on your system.

Software Prerequisites
----------------------

The following commands need to be on your ``PATH``:

1. ``antechamber``, ``parmchk2``, and ``tleap`` (`AmberTools
   <https://ambermd.org/GetAmber.php#ambertools>`_, version 22 or
   higher); the most convenient source is the ``conda-forge`` channel.
2. ``gmx`` or ``gmx_mpi`` (`Gromacs
   <https://manual.gromacs.org/documentation/current/index.html>`_,
   version 2022.1 or higher); available from ``conda-forge``, your
   distribution's package manager, or compiled from source.
3. ``obabel`` (`OpenBabel
   <https://github.com/openbabel/openbabel>`_); preferred installation
   via your Linux distribution's package manager.  OpenBabel is a
   required runtime dependency whenever you let ``htpolynet`` build
   monomer structures from SMILES strings (the recommended workflow —
   see :ref:`molecular_structure_inputs`).  RDKit on its own is not
   sufficient because it has no ``mol2`` writer, so ``htpolynet``
   always shells out to ``obabel`` for the final SDF→mol2 conversion.
   The only way to run ``htpolynet`` without ``obabel`` is to supply
   hand-prepared ``mol2``/``pdb`` files for every monomer.
4. ``dot`` (`Graphviz <https://graphviz.org/>`_); used to render the
   reaction-network plot ``plots/reaction_network.png`` at setup time.
   If ``dot`` is not on the ``PATH`` the build still proceeds; the
   plot is skipped with a warning.

In addition, the in-config SMILES path supports an optional
**atom-mapping** syntax that requires `RDKit <https://www.rdkit.org/>`_.
Install it as an extra::

    $ pip install 'htpolynet[smiles]'

If RDKit is not installed, the index-keyed ``rename_atoms`` form of the
SMILES spec still works.

Installation
------------

Two supported workflows: **uv + Miniforge** (recommended for development
and for any system where you'd like to keep Python and the native MD
binaries cleanly separated), and **conda-only** (simpler one-stop
install if you already manage everything through ``conda-forge``).

uv + Miniforge (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the Python side, use `uv <https://docs.astral.sh/uv/>`_ to manage a
per-project virtual environment.  For the native MD binaries
(AmberTools and Gromacs), use a `Miniforge
<https://github.com/conda-forge/miniforge>`_-managed conda-forge
environment.  Miniforge is community-maintained, uses only the
``conda-forge`` channel, and carries no Anaconda Inc. terms-of-service
encumbrance.

Install uv:

.. code-block:: console

    $ curl -LsSf https://astral.sh/uv/install.sh | sh

Install Miniforge (the installer at
https://github.com/conda-forge/miniforge#install includes one-liners
for every platform), then create the MD-tools environments:

.. code-block:: console

    $ mamba create -n gromacs    -c conda-forge gromacs
    $ mamba create -n ambertools -c conda-forge ambertools parmed

(Splitting AmberTools and Gromacs into separate environments avoids
solver conflicts between their MPI / CUDA-linkage variants.  Combine
them into one env if your platform doesn't trip on that.)

Add the env ``bin`` directories to your ``PATH`` so ``gmx``,
``antechamber``, ``tleap``, and ``parmchk2`` are always available
without needing to activate the envs:

.. code-block:: bash

    # ~/.bashrc (append-only so uv's python isn't shadowed)
    export PATH="$HOME/miniforge3/envs/gromacs/bin:$HOME/miniforge3/envs/ambertools/bin:$PATH"
    export AMBERHOME="$HOME/miniforge3/envs/ambertools"

Install ``htpolynet`` into a uv-managed virtualenv:

.. code-block:: console

    $ git clone git@github.com:AbramsGroup/htpolynet.git
    $ cd htpolynet
    $ uv venv
    $ uv pip install -e '.[smiles]'

For a global ``htpolynet`` command callable from any directory:

.. code-block:: console

    $ uv tool install --editable .

This installs ``htpolynet`` as a uv-managed tool with its own
dedicated environment and a shim in ``~/.local/bin/`` that's available
from any shell.

If your distribution doesn't already provide ``obabel`` and ``dot``,
install them from its package manager.  On Debian/Ubuntu:

.. code-block:: console

    $ sudo apt install openbabel graphviz

On openSUSE / Fedora / RHEL:

.. code-block:: console

    $ sudo zypper install openbabel graphviz    # openSUSE
    $ sudo dnf install openbabel graphviz       # Fedora/RHEL

Conda-only (simpler one-stop install)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using Miniforge (preferred) or any other ``conda``-style installer, you
can manage everything — Python, MD binaries, and ``htpolynet`` itself —
in a single environment:

.. code-block:: console

    $ mamba create -n htpolynet -c conda-forge python ambertools gromacs htpolynet
    $ mamba activate htpolynet

Then install OpenBabel from your distribution's package manager as
described above (OpenBabel via ``conda-forge`` works too but the
distribution package is usually fresher and avoids a Python-extension
ABI link).

PyPI install (Python-only)
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you've installed the native MD binaries some other way, the
Python-only install of ``htpolynet`` from PyPI is:

.. code-block:: console

    $ pip install 'htpolynet[smiles]'

You're responsible for ensuring ``antechamber``, ``parmchk2``,
``tleap``, ``gmx``, and ``obabel`` are reachable on ``PATH``.

Compiling MD tools from source
------------------------------

If your distribution doesn't ship recent enough versions and you don't
want to use a conda-forge build, the native MD tools can be compiled
from source.  Below are reference build recipes.

AmberTools
^^^^^^^^^^

Requires ``csh``, ``flex``, and ``bison``:

.. code-block:: console

    $ tar jxf AmberTools24.tar.bz2
    $ cd amber_src
    $ ./configure --no-X11 --skip-python gnu
    $ source amber.sh
    $ make install

Gromacs
^^^^^^^

Reference CUDA-enabled (single-replica, no MPI) build:

.. code-block:: console

    $ tar xfz gromacs-2025.4.tar.gz
    $ cd gromacs-2025.4
    $ mkdir build
    $ cd build
    $ cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON \
               -DGMX_GPU=CUDA -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs
    $ make
    $ make check
    $ sudo make install

Add to your ``~/.bashrc``:

.. code-block:: bash

    source /usr/local/gromacs/bin/GMXRC

OpenBabel
^^^^^^^^^

Be sure to unpack `Eigen
<https://eigen.tuxfamily.org/index.php?title=Main_Page>`_ first so the
``conformer`` plug-in builds.  Example session where Eigen and
OpenBabel sources are in ``~/Downloads`` and the install prefix is
``~/opt/obabel``:

.. code-block:: console

    $ cd ~/build
    $ tar jxf ~/Downloads/eigen-3.4.0.tar.bz2
    $ tar jxf ~/Downloads/openbabel-3.1.1.tar.bz2
    $ cd openbabel-3.1.1 && mkdir build && cd build
    $ cmake .. -DEIGEN3_INCLUDE_DIR=${HOME}/build/eigen-3.4.0/ \
               -DCMAKE_INSTALL_PREFIX=${HOME}/opt/obabel
    $ make && make test && make install

Then set ``PATH``, ``LD_LIBRARY_PATH``, and ``BABEL_LIBDIR`` to point
at ``${HOME}/opt/obabel``.

Other Prerequisites
-------------------

To use ``htpolynet`` effectively, working knowledge of the following is
helpful:

1. MD simulation in general and Gromacs specifically.
2. The General Amber Force Field (GAFF), including

   a. how to use ``antechamber``, ``tleap``, and ``parmchk2`` to
      generate GAFF parameterizations; and
   b. how to use those parameterizations inside Gromacs.

3. Polymer chemistry, at least for the systems you intend to simulate.
