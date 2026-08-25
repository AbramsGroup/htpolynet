.. _container_usage:

Container Usage
---------------

A pre-built container image is published to the GitHub Container Registry at
``ghcr.io/cameronabrams/htpolynet:latest``.  It bundles Gromacs, AmberTools,
OpenBabel, and ``htpolynet`` itself, so no local installation of any of these
tools is required.

.. note::

  The image is published only to GHCR, not to Docker Hub.  Always refer to it
  by its full path ``ghcr.io/cameronabrams/htpolynet[:tag]``.  A bare reference
  like ``docker run htpolynet ...`` will fail because Docker resolves unqualified
  names against Docker Hub (``docker.io/library/htpolynet``), where no such
  image exists.  If you want a short local alias, tag the pulled image once::

    $ docker pull ghcr.io/cameronabrams/htpolynet:latest
    $ docker tag ghcr.io/cameronabrams/htpolynet:latest htpolynet

Desktop Users (Docker)
^^^^^^^^^^^^^^^^^^^^^^

`Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ (Windows
and Mac) or Docker Engine (Linux) is required.

The recommended way to use the image is with Docker Compose.  Either fetch
the file from the repository::

  $ curl -O https://raw.githubusercontent.com/cameronabrams/htpolynet/main/docker/compose.yml

or save the following as ``compose.yml`` in your working directory:

.. code-block:: yaml

  services:
    htpolynet:
      image: ghcr.io/cameronabrams/htpolynet:latest
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

.. admonition:: Which code is in your image?
   :class: note

   ``htpolynet info`` reports the exact commit the image was built from, which
   is baked in at build time.  This matters because ``:latest`` moves: besides
   moving on every release, it is rebuilt weekly from ``main`` to pick up new
   Gromacs and AmberTools builds, so an image tagged ``:latest`` is frequently
   built from a commit *after* the last release and its version number alone
   will not tell you that.

   For anything whose provenance you need to be able to state later -- a
   published result, a long campaign you want to keep consistent -- pull by
   digest or by the per-commit tag rather than ``:latest``, and record what
   you pulled::

     $ docker pull ghcr.io/cameronabrams/htpolynet@sha256:<digest>

   Every build is also pushed as ``ghcr.io/cameronabrams/htpolynet:<commit-sha>``.

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
      image: ghcr.io/cameronabrams/htpolynet:latest
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

  $ singularity pull htpolynet.sif docker://ghcr.io/cameronabrams/htpolynet:latest

Then run it, binding your working directory:

.. code-block:: console

  $ singularity run --bind $(pwd):/work --pwd /work htpolynet.sif run config.yaml

.. warning::

  **The bundled Gromacs is not a CUDA build.**  The image installs Gromacs
  from conda-forge, whose default package is compiled with OpenCL rather than
  CUDA support, and Gromacs no longer supports OpenCL on NVIDIA devices.  So
  ``--nv`` and ``--gres=gpu:...`` buy you nothing with this image: target a
  CPU partition instead.  ``htpolynet`` detects this at startup, reports
  ``unusable`` in its GPU banner, drops any ``gpu_id`` from the config's
  ``mdrun_options``, and adds ``-nb cpu`` to the ``mdrun`` command line.  If
  you need GPU-accelerated Gromacs on a cluster, run htpolynet natively
  against a CUDA-enabled Gromacs module rather than through this container.

Example shell scripts work the same way as under Docker — the entrypoint will
recognize ``bash`` as an executable and exec it directly:

.. code-block:: console

  $ singularity run --bind $(pwd):/work --pwd /work htpolynet.sif fetch-example 1
  $ singularity run --bind $(pwd):/work --pwd /work htpolynet.sif bash 1-polystyrene.sh --run

Submitting to SLURM
"""""""""""""""""""

Rather than hand-writing a batch script, let htpolynet generate one.  The
:ref:`gen-slurm-script <usage_gen_slurm_script>` subcommand is container-aware:
give it ``--sif`` and it emits a ready-to-submit script that invokes
``htpolynet run`` inside the image.

.. code-block:: console

  $ htpolynet gen-slurm-script config.yaml \
        --sif /shared/containers/htpolynet.sif \
        --job-name htpolynet \
        --partition <cpu-partition> \
        --account <your-account> \
        --nodes 1 --ntasks 1 --cpus-per-task 16 \
        --time 8:00:00 \
        -o submit.sh
  $ sbatch submit.sh

which writes:

.. code-block:: bash

  #!/bin/bash
  #SBATCH --job-name=htpolynet
  #SBATCH --partition=<cpu-partition>
  #SBATCH --account=<your-account>
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=16
  #SBATCH --time=8:00:00

  apptainer exec \
      --bind $(pwd):$(pwd) --pwd $(pwd) \
      /shared/containers/htpolynet.sif \
      htpolynet run config.yaml

Two things to get right when sizing the request:

* **Cores.** Gromacs stops scaling at a few hundred atoms per core, and a
  cure run is dominated by *many short* ``mdrun`` invocations where
  per-invocation startup cost matters more than peak throughput.  Check your
  system size with ``htpolynet input-check config.yaml`` and pick cores
  accordingly — for a 14,000-atom system, 16 cores is already near the knee,
  and a full 48-core node would be slower, not faster.

* **Filesystem.** A cure run does heavy small-file I/O every iteration, so
  submit from a scratch or parallel filesystem rather than an NFS-mounted
  home or group share.  Note that the generated script runs in the submit
  directory and does not stage to node-local scratch, so copy results
  somewhere permanent if your scratch is subject to a purge policy.

If your cluster does not use SLURM, invoke the container directly in whatever
batch script your scheduler wants:

.. code-block:: bash

  SIF=/shared/containers/htpolynet.sif

  singularity run \
      --bind $SLURM_SUBMIT_DIR:/work \
      --pwd /work \
      $SIF run config.yaml -proj next
