# Changelog

All notable changes to htpolynet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- All five bundled examples (0–4) are now self-contained YAMLs using the RDKit atom-mapping path on each constituent, so the user names reactive atoms by chemical identity (e.g. `[CH2:1][CH3:2]`) instead of by obabel's output ordering. The legacy `.sh` and `.tgz` siblings have been removed; `htpolynet fetch-example N` now delivers a single `.yaml` for any N. Example 4 (DFDA/FDE) additionally gets a `reactive_atoms` entry for `O1`/`O2` that was missing in its prior shell script (the cap reaction references them). `htpolynet fetch-example 1` delivers the YAML directly; usage collapses to `htpolynet run 1-polystyrene.yaml`. The legacy `1-polystyrene.sh` and `1-polystyrene.tgz` have been removed. `fetch-example` now prefers `.yaml` > `.sh` > `.tgz`.
- Final-stage save now emits `final.viz.psf` (real bond topology, written via parmed from `final.top` + `final.gro`) and `final.viz.tcl` (drops any bond longer than 3 Å from the display) alongside the existing `final.gro` / `final.top` / `final.tpx` / `final.grx`. Load with `vmd final.viz.psf final.gro -e final.viz.tcl` to view a crosslinked network without the "long bonds across PBC" artifact. The TCL uses `topo getbondlist both` / `topo setbondlist both $list` — the valid topotools 1.x flag values are `type`, `order`, `both`, `none`; the earlier `all` returned an empty list silently and reported "PSF appears to carry no bonds". The TCL also prints the first bond's measured length so the user can verify VMD loaded coordinates in Å.
- New CLI subcommand `htpolynet make-viz` regenerates `final.viz.psf` + `final.viz.tcl` from any `final.top` + `final.gro` pair without re-running the full workflow. Defaults assume the current directory has `final.top` and `final.gro` (i.e. you've `cd`'d into `systems/final-results/`); override with `-top` / `-gro` / `-prefix`.
- `write_top` now casts known-int columns (atom indices, function codes, dihedral periodicities, `nrexcl`, etc.) to pandas' nullable `Int64` before serialization, so they emit as e.g. `2` rather than `2.0`. The float form had been silently accepted by `gmx grompp` but rejected by `parmed`'s gromacs top reader, which broke the new `.viz.psf` generation.
- `gmx --version` output is now parsed for `GPU support:` (CUDA, OpenCL, SYCL, disabled) and shown alongside the version line in the startup banner.
- Consistency check: if the YAML config sets `mdrun_options.gpu_id` but the installed gmx was built without GPU support, or no GPU devices are visible on the host, the option is dropped and a warning is logged. This prevents the runtime crash that `gmx mdrun -gpu_id 0` produces when zero devices are detected.
- Cache hits during parameterization are now logged at INFO ("Using cached parameterization for `<name>`") instead of DEBUG, plus a post-loop summary line tallying reused vs freshly-parameterized molecules and a reminder of the `--force-parameterization --force-checkin` flags to invalidate stale entries. Pairs with the new "Parameterization caching" section in the user-guide.
- The `-restart` flag now emits a prominent runtime warning that resumption is experimental and known to fail at the first cure-stage topology update; the argparse help string is annotated likewise, and `docs/source/user-guide/usage.rst` carries an expanded warning explaining the root cause (in-memory cure state is not fully reconstructible from `cure_state.yaml` + on-disk topology files). Pre-cure stages still resume correctly; this section is parked pending a redesign of cure-state persistence.
- Fixed: `htpolynet run -restart` failed in `CureState.from_yaml` with a `ConstructorError` for the `!!python/object:` tag, because curecontroller's loader was `yaml.FullLoader` (which recent PyYAML tightened to reject Python object tags) while the matching `yaml.dump(self)` writes those tags. Switched to `yaml.Loader` to match what `checkpoint.py` already uses.
- Fixed: rebuild `self.chain_manager` from the reloaded coordinates on restart. `do_initialization` is correctly skipped by the checkpoint decorator on resume, but it's also where `chain_manager` was being constructed, so subsequent stages (`do_cure`) hit `AttributeError`. `do_workflow` now reconstructs it from the loaded TopoCoord whenever a checkpoint payload is present.
- Fixed: `htpolynet run -restart` could die with `shutil.SameFileError` when the userlibrary search fell through to `projPath` and the cwd already lived inside it (so the source file IS the destination). `projectfilesystem.py` now uses a `_safe_copyfile` helper that no-ops when src and dst resolve to the same path.
- Fixed: example 2 HIE constituent's SMILES used `[C:1]` (zero implicit H by SMILES bracket-atom rules) instead of `[CH:1]`, so the α-carbon was emitted at valence 3, antechamber typed it `c2`, and tleap failed with "no angle parameter for o - c2 - os". Documented the bracket-atom H-count gotcha in `docs/source/user-guide/molecular-structure-inputs.rst`.
- Fixed: the RDKit SMILES path now goes through an SDF (molfile) intermediate to obabel rather than PDB. PDB does not carry bond orders, so obabel had to re-infer them and frequently mis-assigned a carbonyl carbon as the alkene sp2 type (`C.2` from a `C-O` single bond rather than `C.2`+`O.2` double-bond pair), which propagated to GAFF as `c2` instead of `c` and broke tleap with "no angle parameter for o - c2 - os" on monomers with ester groups (e.g. HIE in example 2). SDF preserves bond orders, so obabel emits the right sybyl types and antechamber assigns the correct GAFF types.
- Fixed: SMILES-generated mol2 files were being written to `projPath/lib/molecules/inputs/<NAME>.mol2` because `Runtime.__init__` runs after `pfs._setup_project_dir` has `chdir`'d into the project directory. `pfs.checkout()` looks in the *user library* (`rootPath/lib/...`), so the files were unfindable and molecule generation fell through to a `BPA.pdb`/`STY.pdb` assertion. `materialize_smiles_inputs` is now invoked with an absolute `inputs_dir` rooted at the user library, and `htpolynet run` pre-creates `lib/molecules/{inputs,parameterized}/` at startup so the library is wired up even when the user's working directory had no `lib/`.
- Constituents in the YAML config may now carry a `smiles:` key.  When present, htpolynet generates `lib/molecules/inputs/<NAME>.mol2` itself before parameterization, eliminating the obabel/sed boilerplate that example shell scripts have historically duplicated.  Reactive atom names are set via either `rename_atoms: {<1-based-index>: <name>}` (obabel path, always available) or `reactive_atoms: {<smiles-map-num>: <name>}` (RDKit path, used when the SMILES contains `[*:N]` atom-mapping tokens and RDKit is importable).  RDKit is an optional extra: `pip install 'htpolynet[smiles]'`; the container ships it by default.

- New `docker-entrypoint.sh` that auto-detects the host owner of the `/work` bind mount and drops privileges via `gosu` before invoking `htpolynet`. Users no longer need to set `--user`, `HOST_UID`/`HOST_GID`, or any other env vars — output files are written with host ownership automatically. The script also writes an `/etc/passwd` entry for the runtime uid so `gosu` resolves `HOME` to `/home/htpolynet` rather than falling back to `/`.
- Entrypoint dispatches by inspecting the first argument: if it resolves to an executable on `PATH` (`bash`, `python`, `obabel`, ...) it is exec'd directly; otherwise it is treated as an `htpolynet` subcommand. This makes it possible to run `docker compose run --rm htpolynet bash 1-polystyrene.sh --run` (i.e. drive the example shell scripts that themselves call `obabel`/`htpolynet`).

### Changed

- Docker-related files moved from the repo root into a new `docker/` subdirectory: `docker/Dockerfile`, `docker/compose.yml`, `docker/docker-entrypoint.sh`. The GH Actions build workflow and the docs `curl -O` URL are updated to match.
- Dockerfile installs `gosu` and uses the new entrypoint script.
- `compose.yml` simplified: no `user:` field; no `HOST_UID`/`HOST_GID` substitution. The entrypoint handles uid mapping at runtime.
- `compose.yml` bind mount uses `${PWD}:/work:Z` so SELinux-enforcing hosts (Fedora, RHEL, openSUSE Tumbleweed, ...) relabel the host directory to `container_file_t`; without this the container is denied writes regardless of POSIX permissions. Harmless on non-SELinux systems.
- `compose.yml` sets `MPLCONFIGDIR=/tmp/matplotlib` in the environment to silence matplotlib's "not a writable directory" warning.

### Fixed

- Dockerfile: pre-create `/home/htpolynet` with mode `0777` so named docker volumes mounted there inherit a world-writable initial state. Without this, a fresh `htpolynet-home` volume came up owned by root and the non-root container user could not create `~/.htpolynet`, `~/.config/matplotlib`, etc.

### Documentation

- Container-usage page rewritten around the entrypoint-driven uid mapping; covers SELinux + `:Z` and the role of the `htpolynet-home` named volume.

## [2.0.1] - 2026-05-12

### Changed

- `compose.yml`: bind-mount source switched from `.` to `${PWD}` so a single shared `compose.yml` referenced via `docker compose -f` mounts the caller's working directory rather than the file's directory.
- `compose.yml`: `user:` field now reads `${HOST_UID}` / `${HOST_GID}` instead of `${UID}` / `${GID}` — bash's `UID` is read-only and `GID` is not exported, so the original form silently fell back to `0` (root) and broke writes into the bind mount under rootless Docker.
- `compose.yml`: container now gets a persistent `HOME` via a named `htpolynet-home` docker volume — without this, `~/.htpolynet` resolved to `/.htpolynet` (root of the container fs) for the non-root user and the user cache could not be created. The named volume also keeps parameterized monomers/oligomers around across `docker compose run --rm` invocations.

### Fixed

- Dockerfile header comments: removed duplicated `htpolynet` token in the example `docker run` invocations (the `ENTRYPOINT` already provides it) and added `--user $(id -u):$(id -g)` so output files are not owned by root.

### Documentation

- Container-usage page now notes that the image is published only to GHCR (a bare `docker run htpolynet` resolves against Docker Hub and fails) and shows a `docker tag` shortcut for a local alias.
- Added a `curl -O` one-liner for fetching `compose.yml` directly from the repo.

## [2.0.0] - 2026-05-07

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
