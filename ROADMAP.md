# Roadmap

Ideas worth doing that we haven't done yet. This is a living list, not a
commitment or a schedule. When something here gets done, it moves to
`CHANGELOG.md` and comes off this page.

Rough ordering within each section is by value, not by effort.

## Container and deployment

- **CUDA-enabled Gromacs in the container image.** The published image
  installs Gromacs from conda-forge, whose default linux-64 package is
  built with OpenCL (not CUDA) and generic `AVX2_256` SIMD. Gromacs no
  longer drives NVIDIA devices through OpenCL, so the image cannot use a
  GPU at all, and on an AVX-512 host it also leaves single-core
  throughput on the table. For reference, Picotte's own module is
  `AVX_512` + CUDA. Doing this well probably means a second image tag
  (e.g. `:cuda`) built against `gromacs=*=nompi_cuda*` rather than
  changing the default, since it pulls in the CUDA runtime and inflates
  the image substantially, and it needs a GPU-equipped runner or a
  manual build to verify. The CPU image should stay the default so that
  `docker run` on a laptop keeps working. Note that
  `software.gpu_unusable_reasons()` already reasons about this correctly,
  so a CUDA image would simply start passing its checks rather than
  needing new logic.
- **Publish a digest or version tag people can pin.** `:latest` moves
  every week via the scheduled rebuild, so a run recorded as "built with
  the container" is not reproducible. Per-commit tags already exist;
  what's missing is documenting that users should pin one.
- **Retire `ghcr.io/abramsgroup/htpolynet`.** Superseded by the
  `cameronabrams` package; still public and still serving a June image to
  anyone with an old link.

## Testing and CI

Coverage as of the last measurement: **38.8%** overall.

- **`repair/` has no tests at all** — `cyanate_cap.py` (208 statements)
  and `topology_surgery.py` (125), both at 0%. This is the highest-value
  gap: the postcure repair stage makes the strongest correctness claim in
  the project ("atom conservation is exact"), and right now the only
  thing checking it is reading a residue census at the end of a
  multi-hour build. It is pure topology manipulation, so it can be tested
  deterministically in milliseconds against a synthetic `TopoCoord`
  carrying triazines at k=0,1,2,3 — assert atom counts, the residue
  census, cap placement, and that no unreacted bridge -OH survives.
- **An end-to-end example in CI.** A deliberately tiny build (a
  20-molecule, few-ps variant of example 0) run inside the container
  would cover `core/runtime.py` and `cure/curecontroller.py` — 1,056
  statements, both at 0% — in the only way that is honest, since faking
  the whole AmberTools/Gromacs tool chain to unit-test the orchestration
  is a large effort for less confidence.
- **Remaining zero-coverage modules**: `analysis/postsim.py` (203),
  `cli.py` (165), `analysis/analyze.py` (137), `utils/vmd_viz.py` (75),
  `utils/checkpoint.py` (54).
- **`analysis/plot.py` is at 34%** after the smoke-test pass. The
  diagnostics-log parsers (`diagnostics_graphs`, `_token_match`,
  `_parse_data`) are the part most likely to rot silently — they already
  broke once when modules were renamed — and they are testable against a
  small captured log fixture.
- **Coverage reporting in CI**, so the number is visible on a PR rather
  than something we remember to measure by hand.

## Release and distribution

- **Mint a software DOI.** Enable the Zenodo GitHub integration for
  `cameronabrams/htpolynet`, then the next `scripts/release.sh` run
  archives the release automatically. Afterwards, add the concept DOI (not
  the version DOI) as a README badge and an `identifiers` entry in
  `CITATION.cff`. Note that enabling is not retroactive: releases before
  the toggle are not archived. Also note that adding a `.zenodo.json`
  would make Zenodo ignore `CITATION.cff` entirely — only worth doing if
  we need Zenodo-specific fields such as `grants` for funder linkage.
- **Per-minor Python classifiers.** `pyproject.toml` declares only
  `Programming Language :: Python :: 3`, so the PyPI Python badge reads an
  uninformative `python: 3`. Adding `:: 3.10` through `:: 3.13` would make
  it read `3.10 | 3.11 | 3.12 | 3.13`, which CI already verifies at both
  ends.

## Usability

- **`gen-slurm-script` doesn't stage to scratch.** The emitted script
  runs in the submit directory. A cure run does heavy small-file I/O
  every iteration, so on a cluster whose home and group storage are NFS
  that is the wrong place. The cluster-correct pattern is to run in
  node-local or parallel scratch and copy results back, with a `trap` so
  partial results survive a timeout. It also cannot emit a job array,
  which is the right shape for sweeping a set of configs.
- **`-restart` is documented as "EXPERIMENTAL: broken at the cure
  stage".** A build that dies mid-cure currently has to start over — the
  worst possible time to lose work, since cure is the longest stage.
- **Drop the pre-3.5 matplotlib fallback** in `analysis/plot.py`'s
  `_get_cmap()` once `matplotlib>=3.6` is a safe floor; the
  `matplotlib.colormaps` registry is then always present.

## Example depot

- **Example 5 is still not laptop-scale.** The `68d81fb` retune cut the
  monomer pool to make HTPB/IPDI "fit in a reasonable wall time on a
  laptop", but a validation run on 16 CPU cores took **10h17m** — about
  ten times any other example (the next longest, PACM/DGEBA, was 2h15m).
  The cost is structural: densification starts at 10 kg/m³ in a 27 nm box
  and needs all 50 NPT repeats to compress geometrically (a steady 2.9%
  box-side reduction per repeat) up to ~860 kg/m³, and the cure stage then
  does drag/relax on long chains. Worth revisiting whether the low
  starting density is really necessary, or whether the example can start
  denser with a shorter densification.


- **More cyanate-ester variants.** Example 6 covers bisphenol-A
  dicyanate. The same reaction and repair machinery carries over to other
  bisphenol bridges with only a SMILES swap in `constituents` — bisphenol
  F (methylene), bisphenol E (methylethylidene), hexafluorobisphenol A,
  thioether, sulfone, dicyclopentadiene. A homologous series would
  exercise the repair stage across chemistries and give the tutorials a
  structure-property story.
