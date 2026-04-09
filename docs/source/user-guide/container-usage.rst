.. _container_usage:

Container Usage
---------------

A pre-built container image is published to the GitHub Container Registry at
``ghcr.io/abramsgroup/htpolynet:latest``.  It bundles Gromacs, AmberTools,
OpenBabel, and ``htpolynet`` itself, so no local installation of any of these
tools is required.

Desktop Users (Docker)
^^^^^^^^^^^^^^^^^^^^^^

`Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ (Windows
and Mac) or Docker Engine (Linux) is required.

The recommended way to use the image is with Docker Compose.  Save the
following as ``compose.yml`` in your working directory:

.. code-block:: yaml

  services:
    htpolynet:
      image: ghcr.io/abramsgroup/htpolynet:latest
      volumes:
        - .:/work
      working_dir: /work
      user: "${UID:-0}:${GID:-0}"

Then run ``htpolynet`` subcommands via:

.. code-block:: console

  $ docker compose run --rm htpolynet run config.yaml

All ``htpolynet`` subcommands work the same way:

.. code-block:: console

  $ docker compose run --rm htpolynet fetch-example 4
  $ docker compose run --rm htpolynet info
  $ docker compose run --rm htpolynet postsim -proj proj-0 -cfg postsim.yaml

The ``compose.yml`` file mounts the current directory into the container as
``/work`` and runs the process as your host user, so all output files are
written with your own ownership.

.. note::

  On Windows the ``user:`` field in ``compose.yml`` has no effect (Docker
  Desktop on Windows always runs as the current user).  Output files will be
  owned correctly without any changes.

GPU support
"""""""""""

If you have an NVIDIA GPU and the
`NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_
installed, add a ``deploy`` block to your local copy of ``compose.yml``:

.. code-block:: yaml

  services:
    htpolynet:
      image: ghcr.io/abramsgroup/htpolynet:latest
      volumes:
        - .:/work
      working_dir: /work
      user: "${UID:-0}:${GID:-0}"
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: all
                capabilities: [gpu]

``htpolynet`` will detect the available GPU(s) automatically at startup.

HPC Users (Singularity/Apptainer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most HPC clusters provide `Singularity <https://docs.sylabs.io/guides/latest/user-guide/>`_
or `Apptainer <https://apptainer.org/docs/user/latest/>`_ rather than Docker.
Both can pull the image directly from the container registry.

Pull the image once (store it somewhere on shared storage so cluster members
can share it):

.. code-block:: console

  $ singularity pull htpolynet.sif docker://ghcr.io/abramsgroup/htpolynet:latest

Then run it, binding your working directory:

.. code-block:: console

  $ singularity run --bind $(pwd):/work --pwd /work htpolynet.sif run config.yaml

For GPU nodes, add the ``--nv`` flag:

.. code-block:: console

  $ singularity run --nv --bind $(pwd):/work --pwd /work htpolynet.sif run config.yaml

A typical SLURM job script might look like:

.. code-block:: bash

  #!/bin/bash
  #SBATCH --job-name=htpolynet
  #SBATCH --nodes=1
  #SBATCH --ntasks=8
  #SBATCH --gres=gpu:1          # remove if no GPU partition
  #SBATCH --output=slurm-%j.out

  SIF=/shared/containers/htpolynet.sif

  singularity run --nv \
      --bind $SLURM_SUBMIT_DIR:/work \
      --pwd /work \
      $SIF run config.yaml -proj next
