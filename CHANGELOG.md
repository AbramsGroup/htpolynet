# Changelog

All notable changes to htpolynet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-04-15

### Changed

- Package renamed from `HTPolyNet` to `htpolynet` (fully lowercase) for PEP 8 compliance and PyPI consistency.
- Runtime now logs the HTPolyNet git commit hash at startup, with a warning when uncommitted changes are present.

### Added

- Apptainer/Singularity container support: distributed as a `.sif` image for reproducible execution on HPC clusters.
- New `gen-slurm-script` subcommand generates a ready-to-submit SLURM batch script from an htpolynet YAML config file.

### Fixed

- Chain-expansion bug: bond-chain `ChainManager` was not rebuilt for monomers on the fetch path, causing `bondchain_expand_reactions` to produce no chain-extended oligomers in runs that reused cached parameterizations.

## [1.0.9] - 2025-01-01

### Added

- `minimum_bondcycle_length` parameter to allow for cyclic polymerization above a certain threshold length.

### Fixed

- Rings not transferred from monomer templates if they are pre-parameterized.
- Atom indexes in bondchain structure not remapped after atom deletion.

## [1.0.8] - 2024-01-04

### Changed

- Uses `chordless_cycles` to find rings; `ringidx` is no longer a unique atom attribute; improved ring-pierce detection.
