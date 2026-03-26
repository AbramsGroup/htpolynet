# HTPolyNet container image
# Includes Gromacs, AmberTools (antechamber, tleap, parmchk2), and OpenBabel via conda-forge.
#
# Usage:
#   docker run --rm -v $(pwd):/work ghcr.io/abramsgroup/htpolynet htpolynet run config.yaml
#
# To avoid files being written as root, pass your host uid/gid:
#   docker run --rm -v $(pwd):/work --user $(id -u):$(id -g) ghcr.io/abramsgroup/htpolynet htpolynet run config.yaml
#
# GPU support (requires nvidia-container-toolkit):
#   docker run --rm --gpus all -v $(pwd):/work ghcr.io/abramsgroup/htpolynet htpolynet run config.yaml

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
