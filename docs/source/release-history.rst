.. _release-history:

###############
Release History
###############

2.0.0 — 2026-05-07
===================

* Package renamed from ``HTPolyNet`` to ``htpolynet`` (fully lowercase) for PEP 8 compliance and PyPI consistency.
* Apptainer/Singularity container support: HTPolyNet is now distributed as a ``.sif`` image for reproducible execution on HPC clusters without manual dependency installation.
* New ``gen-slurm-script`` subcommand generates a ready-to-submit SLURM batch script from an htpolynet YAML config file.  Key features:

  * Containerized execution via Apptainer by default (``--native`` option for bare-metal installs).
  * GPU awareness: ``--nv`` flag is added automatically when ``gres`` contains ``gpu``.
  * Persistent SLURM settings in a ``slurm:`` section of the config file; CLI flags override them.
  * Pass-through support for non-standard ``#SBATCH`` directives.

* Runtime now logs the HTPolyNet git commit hash at startup, with a warning when uncommitted changes are present.
* Fixed chain-expansion bug: bond-chain ``ChainManager`` was not rebuilt for monomers on the fetch path, causing ``bondchain_expand_reactions`` to produce no chain-extended oligomers in runs that reused cached parameterizations.

1.0.9 *(untagged)*
==================

* ``minimum_bondcycle_length`` parameter added to allow for cyclic polymerization above a certain threshold length.
* Bugfixes:

  * Rings not transferred from monomer templates if they are pre-parameterized.
  * Atom indexes in bondchain structure not remapped after atom deletion.

1.0.8 — 2024-01-04
===================

* Uses ``chordless_cycles`` to find rings; ``ringidx`` is no longer a unique atom attribute; improved ring-pierce detection.

1.0.7.2 *(untagged)*
====================

* Moved Library package to ``resources`` subpackage of ``htpolynet``.

1.0.6 — 2023-06-21
===================

* ``gmx``-style ``analyze`` subcommand added.

1.0.5 — 2022-09-29
===================

* Post-build MD simulations and plotting functionalities added.

1.0.2 — 2022-09-16
===================

* Enhanced molecule-network graph drawing in the ``plot`` subcommand.

1.0.1 — 2022-09-07
===================

* Fixed atom index assignment issue for systems with more than 100,000 atoms.

1.0.0 — 2022-09-03
===================

* First release.

0.0.1 — 2022-08-29
===================

* Initial beta version.
