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
  the image substantially. The CPU image should stay the default so that
  `docker run` on a laptop keeps working. Note that
  `software.gpu_unusable_reasons()` already reasons about this correctly,
  so a CUDA image would simply start passing its checks rather than
  needing new logic.

  Checked 2026-08-25, and the pieces are all there: conda-forge ships
  `gromacs 2026.3` for linux-64 in a `nompi_cuda` variant alongside the
  `nompi` one the image installs today, so the Dockerfile change is a build
  string on the existing `mamba install` line, not a source build.

  **The one real obstacle is the build runner, and it has a known
  workaround.** The CUDA variant depends on `__cuda`, a *virtual* package
  that conda synthesizes only where a CUDA driver is present, so the solve
  fails outright on a GPU-less GitHub Actions runner. Setting
  `CONDA_OVERRIDE_CUDA=12.9` in the build environment satisfies it and lets
  the image build on an ordinary runner. That gets the image *built* but not
  *verified*: whether `mdrun` actually offloads still needs real hardware, so
  the first CUDA image has to be smoke-tested by hand on a GPU node before it
  is advertised. Note the CUDA variant also carries a hard
  `cuda-toolkit >=12.9,<13` dependency, which is where the size goes -- the
  gromacs package alone is 63 MB against 35 MB, and the toolkit is
  gigabytes on top of that.

  Picotte is the obvious place to smoke-test it, since its own Gromacs module
  is `AVX_512` + CUDA and would give a same-hardware comparison.
- **Publish a digest or version tag people can pin.** `:latest` moves
  every week via the scheduled rebuild, so a run recorded as "built with
  the container" is not reproducible. Per-commit tags already exist;
  what's missing is documenting that users should pin one.

- **The image's Gromacs and AmberTools are unpinned, so two images with
  identical htpolynet code can compute different numbers.** `docker/Dockerfile`
  installs `ambertools`, `gromacs`, `parmed` and `rdkit` from conda-forge with
  no version constraints, and the weekly rebuild exists precisely to pick up
  whatever is newest. So the commit stamp now tells you which htpolynet
  produced a build, and still does not tell you which force-field tool chain
  did. Observed live: the image built 2026-08-25 carries AmberTools 26.0 and
  Gromacs 2026.3, while panacea's native environment is on Gromacs
  2025.4 -- a different major version against the same htpolynet.

  Unpinning is deliberate and mostly right: pinning would freeze the image on
  old Gromacs and defeat the point of a weekly rebuild. The gap is that
  nothing *records* what a given image resolved to, so the versions are
  discoverable only by running `htpolynet info` inside it and writing the
  answer down by hand. The better of two candidates is an image
  **label** carrying a `conda list` export, written at build time: `docker
  inspect` then answers the question without running the image, so the record
  survives reaching someone who cannot run the container at all -- a reviewer,
  an archive, a future reader holding only the digest. The alternative, making
  `htpolynet info` machine-readable so a build can capture it, requires the
  image to still be runnable, which is the weaker guarantee. This is the same
  requirement as the build manifest below, one layer further down -- what
  produced this build, all the way to the compilers.

  Raised by the calibration study, which found the 2025.4/2026.3 split only
  because it went looking after an unrelated prompt.

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

- **External services still keyed to the old repo identity.** The Aug 2026
  transfer from `AbramsGroup/HTPolyNet` to `cameronabrams/htpolynet` moved
  the code but left every integration pointing at the old owner. Three
  broke and were fixed during the 2.2.0 release: the GHCR package path
  (docs referenced a package that had never been published under the new
  owner), PyPI trusted publishing (`invalid-publisher` — the claim no
  longer matched, so the v2.2.0 upload failed until the publisher was
  re-registered), and Read the Docs, whose project `repository.url` is
  **still** `https://github.com/AbramsGroup/htpolynet`. Builds succeed
  anyway because GitHub redirects the clone, but tag versions never sync —
  `/en/v2.2.0/` 404s and the API reports "No Version matches the given
  query". Fix the RTD project URL, then activate the tagged version. Worth
  keeping this list as the checklist if the repo ever moves again.


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
- **Make `input-check` a real config linter.** It currently reports only
  the initial atom count. Everything a new config gets wrong is checkable
  cheaply and statically, before the user spends hours discovering it:
  (a) that each `symmetry_equivalent_atoms` group really is topologically
  equivalent — RDKit canonical ranks with `breakTies=False` settle it in
  about ten lines, and a wrong group silently generates reaction templates
  the cure stage will never match; (b) that the A2+B3 site counts actually
  balance at the given monomer `count`s, and that `desired_conversion` is
  reachable given them; (c) that no `reactive_atoms` name collides with the
  bare element names `_reset_names_to_element` assigns to every unmapped
  atom, since the repair stage looks atoms up by name; (d) that the
  requested `charge_method` matches the provenance record of any cached
  parameterization of the same name -- the build itself now rejects a
  mismatch, but reporting it in the pre-flight is cheaper than discovering
  mid-run that half the molecules need re-parameterizing. Building the
  cyanate-ester bridge series meant doing (a) and (b) by hand in a throwaway
  RDKit script, which is exactly the work a user should not have to
  reinvent.
- **Generated topologies cannot be compared byte-wise, because ParmEd
  stamps them.** Every `.top` htpolynet writes opens with a ParmEd header
  recording the invoking user, the host, and the date:

      ;   File TAZ.top  was generated
      ;   By user: cfa (1000)
      ;   On host: panacea.chemeng.drexel.edu
      ;   At date: Fri. May  5 15:55:38 2026

  So two physically identical parameterizations never hash the same, and the
  obvious check -- "did these two builds produce the same parameters for the
  shared monomers?" -- returns a false difference. The calibration study hit
  this on its bridge series and nearly reported the series confounded on the
  strength of an md5; the fix on their side was to strip comment lines before
  comparing, which every user will have to reinvent. Note the header is
  itself provenance, but the wrong kind: it records *when and where* rather
  than *what directives*, which is what the `parm` record now covers, and it
  actively prevents the comparison you would want. The `.itp` files carry no
  such header and the `parm` records are JSON with sorted keys, so both
  already compare cleanly. Worth either normalizing the `.top` header away or
  shipping a comparison helper; this belongs with the manifest entry below,
  since both are about being able to answer "is this the same build?".

- **`-lib` is read-only, and nothing says so.** `pfs.checkout()` and
  `pfs.exists()` consult the `-lib` user library first, then the user cache,
  then the system library -- but `pfs.checkin()` writes unconditionally to the
  user cache, and `UserLibrary` has no `checkin()` method at all. So pointing
  a run at `-lib somewhere` redirects where it *reads* parameterizations from
  while leaving where it *writes* them untouched. That asymmetry is invisible
  from the flag, from `--help`, and from the docs, and it is a natural thing
  to get wrong: a user who passes `-lib` for provenance reasonably concludes
  their products are being contained there. Either `checkin()` should prefer
  the user library when one is configured, or the flag and docs should say
  plainly that it governs lookup only. Verified empirically 2026-08-23: with
  `userlibrary` set, `pfs.checkin()` still landed the file in the user cache.

  What makes this more than a naming problem is that the writes are silent
  and cumulative. Nothing logs them, nothing warns, and a run that is
  quietly depositing molecules into a shared library looks identical to one
  that is not -- the only way to discover it is to audit mtimes, which
  nobody does unprompted. The calibration study ran three sweeps believing
  `-lib` contained it, and would have added roughly seventy entries to
  `~/.htpolynet` across its remaining bridges; it was caught by reading
  `checkin()`, not by anything the tool said. Whatever the fix, a run should
  be able to say where its parameterizations went -- which is the same
  requirement as the build manifest below, approached from the write side
  rather than the read side. The two may well be one item: *where did this
  come from, and what produced it*.

- **No way to run without writing to the user library.** `pfs.checkin()`
  declines to *replace* an entry unless `--force-checkin` is given, but it
  always *writes* one the library does not yet hold, so any run adds every
  molecule name it produces to `~/.htpolynet`. The flag's help text said
  "force check-in of generated parameter files to the system library", which
  reads as though check-in happens only with the flag, and it named the wrong
  library; that wording is fixed, but the missing capability is real. The
  calibration study wanted exactly this: a way to develop against a config
  without its intermediate products accumulating in a library shared with
  other work. Today the only lever is pointing `HTPOLYNET_CACHE` elsewhere,
  which is a blunt instrument because it also hides the entries you *do* want
  to reuse. A `--no-checkin` flag threaded through
  `Runtime._checkin_parameterization()` would cover it in a few lines.

- **Two different defaults for `charge_method`.** `AMBERTOOLS_DEFAULTS` in
  `external/ambertools.py` says `bcc`, which is what a direct
  `GAFFParameterize()` call with no directives gets; `Runtime.runtime_defaults`
  says `gas`, which is what every actual build gets, because
  `_apply_runtime_defaults()` fills it in before AmberTools is ever reached.
  Both were already there and the provenance record is consistent either way
  -- whichever default applies is the one recorded -- so this is a
  readability trap rather than a live bug. It is still worth collapsing to
  one value, and the answer is probably `gas`, since changing what builds
  default to would silently change everyone's charges, which is the exact
  class of harm the record was added to prevent.

- **Nothing durable records that a build reused parameterizations of
  unverified provenance.** The parameterization stage now warns per molecule
  and again in a block at the end of the stage, but both live only in the
  log, and a log is the first thing discarded. Someone reading a result six
  months later -- or a reviewer asking what a published network was actually
  parameterized with -- has no artifact to check. htpolynet writes no build
  manifest today; `profile.json` holds timings only, and the diagnostic log
  is the log. A small `build-manifest.json` in the project directory listing
  each molecule, its origin (`newly parameterized` / `previously
  parameterized`), and its provenance record where one exists would carry
  that, and would be worth more than this one flag: it is also the natural
  home for the config hash, the htpolynet and AmberTools versions, and the
  seed once seeds exist. Raised by the calibration study, which caught the
  original cache bug from a wall-clock anomaly rather than from any log line.
  See also the `-lib` entry above: "a run should be able to say where its
  parameterizations went" and "a result should be able to say what produced
  it" are the same requirement from two directions, and a manifest that
  recorded the check-in destination would answer both.

- **A cached parameterization is not checked against the input structure
  it was built from.** The provenance record added alongside the cache now
  covers `charge_method`, `net_charge` and `atom_type` -- everything in the
  AmberTools invocation -- but not the structure antechamber consumed. So
  editing `lib/molecules/inputs/TAZ.mol2` and re-running still reuses the
  parameterization of the *old* geometry, silently, exactly as the charge
  method used to. Hashing the input would close it, and for a monomer that
  is easy: `Molecule.parameterize()` has the input file in the working
  directory at the moment it runs. Two things stopped it going in with the
  rest: (a) a molecule built by a reaction has no stable input to hash --
  `generate()` writes its mol2 from the merged reactant TopoCoord, whose
  coordinates vary run to run, so hashing it would make every generated
  molecule a permanent cache miss; the hash would have to be recorded only
  for `origin == 'unparameterized'` monomers and compared only when both
  sides carry one. (b) `Runtime._cached_parameterization_mismatch()` runs
  before `generate()` checks the input structure out, and `pfs` has no way
  to resolve a library file to an absolute path without copying it into the
  working directory -- `checkout()` always copies. A `pfs.locate(filename)`
  returning the resolved source path across user library, user cache and
  system library is the missing piece, and is worth having on its own.

- **No seed control anywhere, so a build cannot be reproduced.** Three
  independent sources of randomness are all unseeded: `random.sample` for
  conformer selection (`core/runtime.py`), `np.random.random()` for the
  per-bond probability test (`cure/curecontroller.py`), and every `.mdp`
  sets `gen-vel = yes` without `gen-seed`, so Gromacs picks a pseudo-random
  one per run. Two runs of one config therefore diverge — example 6 gave 57
  vs 60 incomplete triazines on two machines at the same 0.90 conversion.
  That is convenient in one direction (independent replicas of a
  quenched-disorder ensemble come free by re-running) but it means a build
  reported in a paper cannot be reproduced exactly, and a failure seen once
  may not reappear. The fix is a top-level `seed:` that feeds all three:
  seed `random` and `np.random` at startup and write `gen-seed` into every
  generated mdp, with replicas then requested by varying it rather than by
  relying on entropy.
- **Drop the pre-3.5 matplotlib fallback** in `analysis/plot.py`'s
  `_get_cmap()` once `matplotlib>=3.6` is a safe floor; the
  `matplotlib.colormaps` registry is then always present.

## Example depot

- **Example 6's postcure anneal peaks too close to *T*:sub:`g` to relax the
  network, and the fix is not simply "run it longer".** Measured on four
  independent BPA builds: the postcure NPT plateau gives 1.1712 ± 0.0029
  g/cm³ where the same systems melted and slowly re-cooled give 1.1983 ±
  0.0037 g/cm³ -- 2.31% apart, with the plateau 2.6-2.8% below experiment
  and the re-cooled value within 1%. The plateau is an under-relaxed
  structure.

  The mechanism, and the reason the obvious fix is the wrong one: the
  example anneals at a 500 K peak, and the measured *T*:sub:`g` for this
  system is 487.9 K. That is 12.1 K above the glass transition, for 80 ps.
  The Tg ladder that produced the relaxed value spent 15,785 ps above
  *T*:sub:`g`, peaking 112 K above it -- 197x the time, in the regime where
  a crosslinked network actually moves. Lengthening an anneal that sits 12 K
  above *T*:sub:`g` extends a process that barely moves anything, so the
  lever is the peak temperature, not the duration.

  **But raising the peak is unlikely to close the gap entirely.** Two
  structures of the same network were held for 5 ns at a 480 K setpoint
  that thermostatted to 477.1 K -- about 11 K *below* the measured
  *T*:sub:`g`, and far longer than the 80 ps the example spends near it.
  Relaxation there is fast at first and then stops: about 87% of an
  initial 34.8 kg/m³ density gap -- the gap measured over the first 200 ps
  -- closes within 3 ns, and the remaining 4.25 kg/m³ persists at
  7.2 sigma, with a last-half trend of +0.030 kg/m³/ns, i.e. not decaying.
  (State the baseline and the window whenever this number is quoted: the
  same data give 83.5-87.8% across the defensible choices of each, so a
  bare percentage is not a fact.)  So heat and time buy most of the
  relaxation quickly and then buy nothing, and a residual difference
  survives that a longer anneal at this scale does not appear able to
  remove.  Whatever replaces the current anneal should
  therefore be argued as a large improvement, not as a fix; an entry
  promising equilibration would oversell what was measured.

  Note what that hold also says about *T*:sub:`g` itself: **relaxation does
  not switch off at the glass transition**, it slows continuously through
  it.  A transition 20-34 K wide by experiment means 11 K below
  *T*:sub:`g` is inside the transition, not deep in the glass, so fast
  early relaxation there is expected rather than anomalous.  This does not
  soften the argument above -- the 197x figure is unchanged and a peak well
  clear of the transition is still the lever -- but it does mean *T*:sub:`g`
  is a soft kinetic boundary for this material, which any protocol
  expressed relative to it has to accommodate.

  **Untested.** Nobody has yet built with a hotter anneal and compared it
  against the melt-and-recool value, which is the experiment that would
  settle it; the argument above is mechanism, not measurement. That is why
  the example is documented rather than changed. Raising the peak also costs
  wall-clock in a tutorial that should stay small, so the test needs to
  report both whether it works and what it costs.

  The general problem is worth more than the instance: **any protocol whose
  relaxation depends on being above *T*:sub:`g` is fragile when *T*:sub:`g`
  is unknown at configuration time**, which it always is -- you cannot know
  it before building the thing. A fixed absolute peak temperature is a guess
  that happens to be right or wrong per chemistry, silently. Something
  expressed relative to an estimated or measured *T*:sub:`g`, or a postcure
  that measures *T*:sub:`g` and then anneals accordingly, would be robust
  where an absolute number is not. That is a real design change, not a
  config edit.

  Raised by the calibration study, which had already published the
  pessimistic density before catching it -- the failure mode is reporting a
  protocol artifact as a force-field result.

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
