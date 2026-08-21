# HTPolyNet container image
# Includes Gromacs, AmberTools (antechamber, tleap, parmchk2), and OpenBabel
# via conda-forge.
#
# The entrypoint auto-detects the host owner of the /work bind mount and drops
# privileges via gosu before invoking htpolynet, so the caller does not need
# --user, HOST_UID/HOST_GID env vars, or a matching /etc/passwd entry.
#
# Build context must be the repo root (this Dockerfile lives in docker/):
#
#   docker build -f docker/Dockerfile -t ghcr.io/cameronabrams/htpolynet:latest .
#
# Recommended usage is via Docker Compose; see docker/compose.yml and
# docs/source/user-guide/container-usage.rst for the full story (incl. GPU
# and Singularity/Apptainer on HPC).
#
#   docker compose run --rm htpolynet run config.yaml
#
# Raw `docker run` equivalent:
#
#   docker run --rm -v $(pwd):/work ghcr.io/cameronabrams/htpolynet run config.yaml
#
# GPU support (requires nvidia-container-toolkit):
#
#   docker run --rm --gpus all -v $(pwd):/work \
#       ghcr.io/cameronabrams/htpolynet run config.yaml

FROM condaforge/miniforge3:latest

# Links the published GHCR package to its source repository.
LABEL org.opencontainers.image.source="https://github.com/cameronabrams/htpolynet"

# Install Gromacs and AmberTools (latest available on conda-forge).
# Miniforge ships with mamba for fast solves; conda-forge is the only
# channel.  No Anaconda Inc. ToS dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openbabel \
        graphviz \
        gosu \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mamba install -y -n base -c conda-forge \
        ambertools \
        gromacs \
        parmed \
        rdkit \
    && mamba clean -afy

# Install HTPolyNet
COPY . /htpolynet
WORKDIR /htpolynet
RUN pip install --no-cache-dir .

# Pre-create HOME for the unprivileged user the entrypoint drops to.  When
# compose.yml mounts a named volume here, Docker initializes the fresh volume
# from this directory and preserves the 0777 mode.
RUN mkdir -p /home/htpolynet && chmod 0777 /home/htpolynet

COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# User data is mounted here at runtime
WORKDIR /work

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]
