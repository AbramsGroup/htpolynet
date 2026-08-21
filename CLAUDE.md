# Working on htpolynet

## Roadmap and changelog

`ROADMAP.md` holds upgrades we have identified but not done. `CHANGELOG.md`
holds what shipped, in Keep-a-Changelog form with a live `[Unreleased]`
section.

Keep both current as a matter of course, without being asked:

- When we decide *not* to do something now — a deferred fix, an idea worth
  keeping, a limitation we chose to live with — add it to `ROADMAP.md` with
  enough context to act on it months later. An entry that just says "make
  Gromacs faster" is worthless; say which build, why it is slow, and what
  the tradeoff is.
- When a roadmap item ships, delete it from `ROADMAP.md` and describe it in
  `CHANGELOG.md` under `[Unreleased]`.
- Surface relevant roadmap items unprompted when we touch related code.

## Running the tests

```
uv run --extra test pytest tests/unit -q
```

`dev` is an alias for the `test` extra; both work.

Three test modules shell out to external binaries — `test_parameterize_react`
needs antechamber/parmchk2/tleap/gmx, and the two `test_gromacs_*` modules
need gmx. They skip when those are absent rather than failing, which is what
lets CI run on a plain runner. With the tool chain present the suite takes a
couple of minutes; without it, about four seconds.

## The test suite runs inside the source tree

`tests/conftest.py` has an autouse `change_test_dir` fixture that chdirs each
test into `tests/unit/<module_name>/` if that directory exists, otherwise
`tests/unit/`. Both are inside the repo. So a test that writes a file to the
working directory litters the source tree, and cleanup that runs only on the
success path leaves the file behind whenever the test fails or is
interrupted. Write scratch files to `tmp_path` or a
`tempfile.TemporaryDirectory` instead.

## Docs must build with zero warnings

```
uv run --with-requirements docs/requirements.txt --with sphinx \
    python -m sphinx -b html docs/source /tmp/docbuild
```

It currently builds clean; keep it that way. Two things that historically
broke it:

- `docs/source/htpolynetpackage.rst` is a hand-maintained list of
  `automodule` directives, so it drifts silently as modules are added,
  moved, or deleted. It once autodoc'd `htpolynet.driver` for months after
  that module ceased to exist. When you add or move a module, add it here.
- Adding a module to that page renders its docstrings for the first time,
  which surfaces latent RST errors. The usual one is a bullet list with no
  blank line before it, which docutils rejects.

## Releases

Use `scripts/release.sh <version>`. It rotates `[Unreleased]` into a dated
section, bumps the version in `pyproject.toml`, commits, tags, and pushes.
Pushing the tag is what triggers publication: `release.yaml` builds and
publishes to PyPI, creates a GitHub Release from the changelog notes, and
kicks a Read the Docs build. The conda-forge autotick bot then opens a
feedstock PR.

Do not hand-roll any of that. The script's preflight also checks that
`pyproject.toml`'s runtime dependencies still match the conda-forge feedstock
recipe — the autotick bot only bumps version and sha, so a dependency change
we forget to mirror ships a broken conda package.

Because the changelog notes become the public release body, keep
`[Unreleased]` free of internal bookkeeping ("got bundled into commit
abc1234") and file entries under the right heading.

## The container

The image is `ghcr.io/cameronabrams/htpolynet`, built by `docker.yml` from
`docker/Dockerfile`. It rebuilds weekly on a schedule and on any tag matching
`v*` or `d*`. A `d*` tag is the way to rebuild the image without cutting a
release — useful when only the Dockerfile changed.

Its Gromacs comes from conda-forge, which means **OpenCL, not CUDA**, and
generic `AVX2_256` SIMD. Gromacs no longer drives NVIDIA devices through
OpenCL, so the image cannot use a GPU: on a cluster, target CPU partitions
and do not request `--gres=gpu` or pass `--nv`. See `ROADMAP.md`.

## Two invariants worth not breaking

- **Colormaps go through `analysis.plot._get_cmap()`**, never
  `matplotlib.cm.get_cmap` directly. The latter was removed in matplotlib
  3.11, and since `pyproject.toml` floors matplotlib without a ceiling, a
  direct call breaks every plot on a fresh install — after densification has
  already spent its compute.
- **GPU usability is decided by `external.software.gpu_unusable_reasons()`**,
  which is the single place that reconciles detected hardware against what
  the gmx build can actually drive. Do not add ad-hoc `if gpu_ids:` checks
  elsewhere; a weaker hardware-only duplicate of this test used to live in
  `grompp_and_mdrun` and disagreed with it.
