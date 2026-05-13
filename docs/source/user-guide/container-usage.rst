.. _container_usage:

Container Usage
---------------

A pre-built container image is published to the GitHub Container Registry at
``ghcr.io/abramsgroup/htpolynet:latest``.  It bundles Gromacs, AmberTools,
OpenBabel, and ``htpolynet`` itself, so no local installation of any of these
tools is required.

.. note::

  The image is published only to GHCR, not to Docker Hub.  Always refer to it
  by its full path ``ghcr.io/abramsgroup/htpolynet[:tag]``.  A bare reference
  like ``docker run htpolynet ...`` will fail because Docker resolves unqualified
  names against Docker Hub (``docker.io/library/htpolynet``), where no such
  image exists.  If you want a short local alias, tag the pulled image once::

    $ docker pull ghcr.io/abramsgroup/htpolynet:latest
    $ docker tag ghcr.io/abramsgroup/htpolynet:latest htpolynet

Desktop Users (Docker)
^^^^^^^^^^^^^^^^^^^^^^

`Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ (Windows
and Mac) or Docker Engine (Linux) is required.

The recommended way to use the image is with Docker Compose.  Either fetch
the file from the repository::

  $ curl -O https://raw.githubusercontent.com/AbramsGroup/htpolynet/main/docker/compose.yml

or save the following as ``compose.yml`` in your working directory:

.. code-block:: yaml

  services:
    htpolynet:
      image: ghcr.io/abramsgroup/htpolynet:latest
      volumes:
        - ${PWD}:/work:Z
        - htpolynet-home:/home/htpolynet
      working_dir: /work
      environment:
        - HOME=/home/htpolynet
        - MPLCONFIGDIR=/tmp/matplotlib

  volumes:
    htpolynet-home:

The container starts as root, and the image's entrypoint script auto-detects
the host owner of ``/work`` and drops privileges (via ``gosu``) before running
``htpolynet``.  This means output files land in your working directory with
your own ownership — no ``--user``, no ``HOST_UID`` / ``HOST_GID`` env vars,
no entries to add to ``~/.bashrc``.

The named ``htpolynet-home`` volume gives the container a writable ``HOME``
for caches that persist across runs.  Most importantly this is where
``~/.htpolynet`` (parameterized monomers, oligomers, etc.) lives — without
this, each ``docker compose run --rm`` would re-run antechamber/tleap from
scratch.  Run ``docker volume rm htpolynet-home`` to wipe the cache.

Using ``${PWD}`` (rather than ``.``) means the mount follows your current
working directory even when you pass ``-f`` to point at a ``compose.yml``
that lives elsewhere::

  $ docker compose -f /path/to/htpolynet/compose.yml run --rm htpolynet run config.yaml

.. note::

  **SELinux hosts (Fedora, RHEL, CentOS, openSUSE Tumbleweed, ...).** The
  ``:Z`` suffix on the ``${PWD}:/work`` mount tells Docker to relabel the host
  directory with a ``container_file_t`` SELinux type so the container can
  write to it.  Without ``:Z`` on an enforcing host, every write fails with
  *Permission denied* regardless of POSIX ownership.  ``:Z`` is harmless on
  systems without SELinux.  Check with ``getenforce`` — if it says
  ``Disabled`` or ``Permissive`` you don't need it, but leaving it in does
  no damage.

The ``MPLCONFIGDIR`` line redirects matplotlib's font/style cache to ``/tmp``
so it doesn't try (and noisily fail) to populate ``~/.config/matplotlib``
inside the container.

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

Running example shell scripts
"""""""""""""""""""""""""""""

The example scripts fetched via ``fetch-example`` call out to ``obabel`` and
``htpolynet`` — both of which live in the container, not on the host.  The
entrypoint dispatches on the first argument: if it resolves to an executable
on ``PATH`` (``bash``, ``python``, ``obabel``, ...), it is run directly;
otherwise it is treated as an ``htpolynet`` subcommand.  So:

.. code-block:: console

  $ docker compose run --rm htpolynet fetch-example 1                   # download self-contained YAML
  $ docker compose run --rm htpolynet run 1-polystyrene.yaml             # launch htpolynet end-to-end

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
        - ${PWD}:/work:Z
        - htpolynet-home:/home/htpolynet
      working_dir: /work
      environment:
        - HOME=/home/htpolynet
        - MPLCONFIGDIR=/tmp/matplotlib
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: all
                capabilities: [gpu]

  volumes:
    htpolynet-home:

``htpolynet`` will detect the available GPU(s) automatically at startup.

HPC Users (Singularity/Apptainer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most HPC clusters provide `Singularity <https://docs.sylabs.io/guides/latest/user-guide/>`_
or `Apptainer <https://apptainer.org/docs/user/latest/>`_ rather than Docker.
Both can pull the image directly from the container registry.

.. note::

  None of the Docker-side machinery (the SELinux ``:Z`` label, the
  ``htpolynet-home`` named volume, the entrypoint's gosu-based uid drop) is
  needed under Apptainer/Singularity.  Those runtimes already run containers
  as the calling user and bind-mount the host's ``HOME`` and ``/etc/passwd``,
  so the image entrypoint sees a non-root uid and falls through to a plain
  ``exec`` of the requested command.  The htpolynet user cache lands in your
  host ``~/.htpolynet`` (no named volume required), and output files in your
  ``--bind`` mount are written with your own ownership.

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

Example shell scripts work the same way as under Docker — the entrypoint will
recognize ``bash`` as an executable and exec it directly:

.. code-block:: console

  $ singularity run --bind $(pwd):/work --pwd /work htpolynet.sif fetch-example 1
  $ singularity run --bind $(pwd):/work --pwd /work htpolynet.sif bash 1-polystyrene.sh --run

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
