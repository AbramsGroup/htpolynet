# HTPolyNet container image
# Includes Gromacs, AmberTools (antechamber, tleap, parmchk2), and OpenBabel via conda-forge.
# ENTRYPOINT is `htpolynet`, so anything after the image name is passed as
# subcommand/arguments (e.g. `run config.yaml`, `fetch-example 4`, `info`).
#
# Recommended usage is via Docker Compose; see compose.yml at the repo root and
# docs/source/user-guide/container-usage.rst for the full story (incl. GPU and
# Singularity/Apptainer on HPC).
#
#   docker compose run --rm htpolynet run config.yaml
#
# Raw `docker run` equivalent (mounts cwd to /work, runs as host user so output
# files are not owned by root):
#
#   docker run --rm -v $(pwd):/work --user $(id -u):$(id -g) \
#       ghcr.io/abramsgroup/htpolynet run config.yaml
#
# GPU support (requires nvidia-container-toolkit):
#
#   docker run --rm --gpus all -v $(pwd):/work --user $(id -u):$(id -g) \
#       ghcr.io/abramsgroup/htpolynet run config.yaml

FROM continuumio/miniconda3:latest

# Install Gromacs and AmberTools (latest available on conda-forge)
RUN apt-get update && apt-get install -y --no-install-recommends \
        openbabel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda install -y -c conda-forge \
        ambertools \
        gromacs \
        parmed \
    && conda clean -afy

# Install HTPolyNet
COPY . /htpolynet
WORKDIR /htpolynet
RUN pip install --no-cache-dir .

# User data is mounted here at runtime
WORKDIR /work

ENTRYPOINT ["htpolynet"]
CMD ["--help"]
