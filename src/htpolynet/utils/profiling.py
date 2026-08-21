"""Run-time profiling: stage wall-clock + subprocess attribution.

A ``RunProfile`` is set as the active profile at the start of a run.
Code wraps interesting sections in ``with profiling.stage('name')`` and
the subprocess wrapper in :mod:`htpolynet.external.command` calls
:func:`record_subprocess` for every external invocation it dispatches.
The active stage is taken from a stack, so subprocess time gets
attributed to whichever stage is innermost when the call happens.

At end-of-run the report is written to the log and to ``profile.json``
beside ``final.top``.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_ACTIVE: Optional["RunProfile"] = None


@dataclass
class _StageFrame:
    name: str
    depth: int
    t_start: float
    enter_seq: int
    self_subprocess: dict = field(default_factory=lambda: defaultdict(float))


@dataclass
class _StageRecord:
    path: str
    depth: int
    enter_seq: int
    wall_seconds: float
    self_subprocess: dict


@dataclass
class RunProfile:
    """A flat record of stage entries paired with subprocess attribution."""
    records: list = field(default_factory=list)
    _stack: list = field(default_factory=list)
    _started_at: float = field(default_factory=time.monotonic)
    _wall_total: float = 0.0
    _next_seq: int = 0

    def enter(self, name: str) -> None:
        depth = len(self._stack)
        self._stack.append(_StageFrame(
            name=name, depth=depth, t_start=time.monotonic(),
            enter_seq=self._next_seq,
        ))
        self._next_seq += 1

    def exit(self) -> None:
        if not self._stack:
            return
        frame = self._stack.pop()
        elapsed = time.monotonic() - frame.t_start
        path_parts = [f.name for f in self._stack] + [frame.name]
        self.records.append(_StageRecord(
            path='/'.join(path_parts),
            depth=frame.depth,
            enter_seq=frame.enter_seq,
            wall_seconds=elapsed,
            self_subprocess=dict(frame.self_subprocess),
        ))
        self._wall_total = time.monotonic() - self._started_at

    def record_subprocess(self, kind: str, duration: float) -> None:
        if self._stack:
            self._stack[-1].self_subprocess[kind] += duration
        else:
            # outside any tracked stage; bucket into an "ambient" pseudo-frame
            self.records.append(_StageRecord(
                path='(ambient)', depth=0,
                wall_seconds=duration,
                self_subprocess={kind: duration},
            ))

    def aggregate_subprocess(self) -> dict:
        agg: dict = defaultdict(float)
        for r in self.records:
            for k, v in r.self_subprocess.items():
                agg[k] += v
        return dict(agg)

    def format_report(self) -> str:
        total_wall = self._wall_total or (time.monotonic() - self._started_at)
        agg = self.aggregate_subprocess()
        total_sp = sum(agg.values())

        lines = []
        lines.append('=' * 78)
        lines.append('Run profile')
        lines.append('=' * 78)
        lines.append(f'Total wall time: {_fmt_dur(total_wall)}')
        lines.append('')
        lines.append(f'{"Stage":<48}{"wall":>12}{"subprocess":>16}')
        lines.append('-' * 78)
        # Render in enter order so parents appear before their children.
        for r in sorted(self.records, key=lambda x: x.enter_seq):
            name = ('  ' * r.depth) + r.path.split('/')[-1]
            self_sp = sum(r.self_subprocess.values())
            lines.append(f'{name[:48]:<48}{_fmt_dur(r.wall_seconds):>12}{_fmt_dur(self_sp):>16}')
        lines.append('-' * 78)
        lines.append('')
        lines.append('Subprocess time aggregated by kind:')
        if agg:
            for kind in sorted(agg, key=lambda k: -agg[k]):
                pct = 100.0 * agg[kind] / total_wall if total_wall else 0.0
                lines.append(f'  {kind:<24}{_fmt_dur(agg[kind]):>10}   ({pct:5.1f}% of wall)')
            pct = 100.0 * total_sp / total_wall if total_wall else 0.0
            lines.append(f'  {"total":<24}{_fmt_dur(total_sp):>10}   ({pct:5.1f}% of wall)')
        else:
            lines.append('  (no subprocess calls recorded)')
        lines.append('=' * 78)
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        total_wall = self._wall_total or (time.monotonic() - self._started_at)
        return {
            'total_wall_seconds': total_wall,
            'aggregate_subprocess_seconds': self.aggregate_subprocess(),
            'stages': [
                {
                    'path': r.path,
                    'depth': r.depth,
                    'wall_seconds': r.wall_seconds,
                    'self_subprocess_seconds': r.self_subprocess,
                }
                for r in sorted(self.records, key=lambda x: x.enter_seq)
            ],
        }

    def write_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def _fmt_dur(seconds: float) -> str:
    if seconds < 1.0:
        return f'{seconds*1000:.0f} ms'
    if seconds < 60.0:
        return f'{seconds:.2f} s'
    m, s = divmod(seconds, 60.0)
    if m < 60:
        return f'{int(m)}m{s:04.1f}s'
    h, m = divmod(m, 60.0)
    return f'{int(h)}h{int(m):02d}m{s:04.1f}s'


def set_active(profile: Optional[RunProfile]) -> None:
    """Install ``profile`` as the active profile (or clear with ``None``)."""
    global _ACTIVE
    _ACTIVE = profile


def current() -> Optional[RunProfile]:
    return _ACTIVE


@contextmanager
def stage(name: str):
    """Context manager: enter a named stage on the active profile, if any."""
    if _ACTIVE is None:
        yield
        return
    _ACTIVE.enter(name)
    try:
        yield
    finally:
        _ACTIVE.exit()


def record_subprocess(kind: str, duration: float) -> None:
    """Attribute ``duration`` seconds of external work to the active stage."""
    if _ACTIVE is not None:
        _ACTIVE.record_subprocess(kind, duration)


def classify_command(command: str) -> str:
    """Classify a shell command string into a profile category.

    Aims to be coarse but informative:

    - ``gmx mdrun ...`` → ``gmx-mdrun``; ``gmx grompp ...`` → ``gmx-grompp``;
      similarly for other subcommands.
    - ``antechamber``, ``parmchk2``, ``tleap``, ``obabel`` → kept as-is.
    - Everything else → the first token of the command.
    """
    if not command:
        return 'other'
    tokens = command.strip().split()
    if not tokens:
        return 'other'
    head = tokens[0].rsplit('/', 1)[-1]
    if head in ('antechamber', 'parmchk2', 'tleap', 'obabel', 'wc', 'split'):
        return head
    # The gmx binary is sometimes invoked through wrappers like
    # ``gmx_mpi`` or ``/opt/gromacs/bin/gmx``; treat anything starting with
    # ``gmx`` as a gromacs call and look for the subcommand.
    if head.startswith('gmx'):
        for t in tokens[1:]:
            if t.startswith('-'):
                continue
            return f'gmx-{t}'
        return 'gmx'
    return head
