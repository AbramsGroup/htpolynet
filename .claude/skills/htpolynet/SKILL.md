---
name: htpolynet
description: Build crosslinked polymer systems with htpolynet — authoring a run configuration, parameterizing monomers, running a cure, and reading the results. Use when the task involves an htpolynet YAML config, the `htpolynet` command (run, input-check, gen-slurm-script, plots, postsim, analyze, make-viz), a monomer's active form or sacrificial hydrogens, a cure that stalls below its target conversion, or sizing an htpolynet build for a cluster.
---

# Working with htpolynet

**Read `docs/source/user-guide/building-a-system.rst` first.** It is the
procedure, start to finish, and it is maintained alongside the code. This
file exists to route you there and to carry the few things that are easy to
get wrong before you get that far. Do not restate the docs here; when they
and this file disagree, the docs are right.

## Start from an example, never from an empty file

`htpolynet fetch-example <n>` unpacks a working configuration. Pick the one
closest in **reaction topology** — chain growth, A2+B3 step growth,
cyclotrimerization — not the one closest in chemistry. Adapt it. Writing a
config from scratch is the slowest route and the one that produces silent
errors.

Run the example unmodified once before editing, to confirm the toolchain and
to get a known-good log to diff against.

## Monomers are described in their active form

The most common conceptual error, and it produces a build that completes and
is wrong rather than one that fails. htpolynet conserves valence: bonding
atoms each give up a sacrificial hydrogen, so you describe the monomer with
its reactive sites already saturated. Styrene is described as ethylbenzene.
See `docs/source/user-guide/molecular-structure-inputs.rst`.

In SMILES bracket atoms the hydrogen count is explicit: `[C:1]` means *zero*
implicit hydrogens. For an sp³ carbon you almost always want `[CH2:1]` or
`[CH3:1]`. A wrong count usually surfaces much later as a missing GAFF angle
parameter in `tleap`.

## Subcommand routing

| Intent | Command |
|---|---|
| Build a system | `htpolynet run <cfg>` |
| Size it before queueing | `htpolynet input-check <cfg>` |
| Stop after parameterization | `htpolynet run --param-only <cfg>` |
| Submit to a cluster | `htpolynet gen-slurm-script <cfg>` |
| Plot a finished build | `htpolynet plots build --proj <dir>` |
| Post-build MD / analysis | `htpolynet postsim`, `htpolynet analyze` |
| Rebuild VMD viz files | `htpolynet make-viz` |
| Report the environment | `htpolynet info` |

## Non-obvious facts

- **The container image cannot use a GPU.** Its Gromacs is conda-forge,
  built against OpenCL, which Gromacs no longer uses for NVIDIA devices.
  Target CPU partitions; never suggest `--gres=gpu` or `--nv`.
- **Size core requests from `htpolynet input-check`**, which reports the
  initial atom count.
- **A cure stalling below its target conversion** is usually a reaction
  template that never matches — suspect `symmetry_equivalent_atoms` and
  reactive-atom names before suspecting the chemistry.
- **Builds are not reproducible run to run.** There is no seed control, so
  replicas come free but an exact rebuild does not. Record the commit.

## Do not

- Do not write workarounds into this file. If the tool surprises a user in a
  way that requires a ritual to avoid, that is a bug: fix it in code, or file
  it in `ROADMAP.md` with enough context to act on later. A skill that
  teaches people to route around a defect keeps the defect.
- Do not hand-roll a release, a version bump, or a `CHANGELOG` rotation; see
  `CLAUDE.md`.
