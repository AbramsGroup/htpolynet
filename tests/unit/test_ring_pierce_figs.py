"""Documentation figures for ring-pierce checking.

Running this module with ``--gen-figs`` writes two PNG files to
``docs/source/user-guide/pics/ring_pierce/``:

    ring_pierce_cases.png   — 2×2 panel: the four geometry scenarios
    ring_pierce_linkcell.png — top-down linkcell neighbourhood illustration

All tests still carry real assertions so they also function as regression
checks when --gen-figs is omitted (figures are just skipped).

Usage::

    pytest tests/unit/test_ring_pierce_figs.py --gen-figs -v

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import pytest
import numpy as np
import pandas as pd

# Skip the entire module if matplotlib is absent
mpl = pytest.importorskip('matplotlib')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from htpolynet.geometry.ring import Ring, RingList
from htpolynet.geometry.linkcell import Linkcell
from htpolynet.geometry.matrix4 import Matrix4


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _regular_polygon(nsides, radius, center=(0., 0., 0.)):
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, nsides, endpoint=False)
    return np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
        np.full(nsides, cz),
    ])


def _atom_df(positions, start_idx=1):
    N = len(positions)
    return pd.DataFrame({
        'globalIdx': list(range(start_idx, start_idx + N)),
        'posX': positions[:, 0],
        'posY': positions[:, 1],
        'posZ': positions[:, 2],
        'linkcell_idx': -1,
    })


def _ring_from_pts(pts, start_idx=1):
    n = len(pts)
    r = Ring(list(range(start_idx, start_idx + n)))
    df = pd.DataFrame({
        'globalIdx': list(range(start_idx, start_idx + n)),
        'posX': pts[:, 0], 'posY': pts[:, 1], 'posZ': pts[:, 2],
    })
    r.injest_coordinates(df)
    return r


# ---------------------------------------------------------------------------
# 3-D drawing helpers
# ---------------------------------------------------------------------------

def _draw_ring(ax, pts, *, fc='steelblue', alpha=0.25, ec='navy', lw=1.5):
    poly = Poly3DCollection([pts], alpha=alpha,
                             facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_collection3d(poly)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               color=ec, s=30, depthshade=False, zorder=5)


def _draw_bond(ax, p1, p2, *, color='crimson', lw=2.5):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            '-', color=color, lw=lw, solid_capstyle='round')
    ax.scatter(*p1, color=color, s=60, depthshade=False, zorder=6)
    ax.scatter(*p2, color=color, s=60, depthshade=False, zorder=6)


def _mark_pierce(ax, pt):
    ax.scatter(*pt, color='gold', s=180, marker='*',
               edgecolors='darkorange', linewidth=1,
               depthshade=False, zorder=10)


def _style(ax, title, lims, zlims):
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_zlim(zlims)
    ax.set_xlabel('x', labelpad=1, fontsize=8)
    ax.set_ylabel('y', labelpad=1, fontsize=8)
    ax.set_zlabel('z', labelpad=1, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, pad=5)
    ax.view_init(elev=25, azim=-55)


def _autostyle(ax, title, pts_list):
    all_pts = np.vstack(pts_list)
    mid = (all_pts.max(0) + all_pts.min(0)) / 2
    r = (all_pts.max(0) - all_pts.min(0)).max() / 2 * 1.15
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_xlabel('x', labelpad=1, fontsize=8)
    ax.set_ylabel('y', labelpad=1, fontsize=8)
    ax.set_zlabel('z', labelpad=1, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, pad=5)
    ax.view_init(elev=25, azim=-55)


# ---------------------------------------------------------------------------
# Figure 1 — geometry cases (2 × 2 panels)
# ---------------------------------------------------------------------------

class TestRingPierceCasesFigure:
    """Generates ring_pierce_cases.png showing four geometry scenarios.

    The underlying assertions are always evaluated (regression value even
    without --gen-figs).  Figure writing is skipped unless --gen-figs is set.
    """

    NSIDES = 6
    RADIUS = 1.5

    # -- shared ring ---------------------------------------------------------

    @pytest.fixture(autouse=True)
    def _ring(self):
        self.pts = _regular_polygon(self.NSIDES, self.RADIUS)
        self.ring = _ring_from_pts(self.pts)

    # -- scenario helpers (also serve as sanity assertions) ------------------

    def _case_pierce_axial(self):
        """Bond along z-axis through ring centre → pierces."""
        B = np.array([[0., 0., -1.8], [0., 0., 1.8]])
        did, pt = self.ring.pierced_by(B)
        assert did, 'axial bond should pierce'
        return B, pt

    def _case_no_pierce_same_side(self):
        """Both bond endpoints above z=0 → no pierce."""
        B = np.array([[-0.5, 0., 0.5], [0.5, 1.0, 0.5]])
        did, _ = self.ring.pierced_by(B)
        assert not did, 'same-side bond should not pierce'
        return B

    def _case_no_pierce_outside(self):
        """Bond crosses ring plane but outside the polygon → no pierce."""
        off = self.RADIUS * 1.5
        B = np.array([[off, off, -1.8], [off, off, 1.8]])
        did, _ = self.ring.pierced_by(B)
        assert not did, 'outside bond should not pierce'
        return B

    def _case_pierce_tilted(self):
        """Same piercing bond after arbitrary rotation + translation → still pierces."""
        M = (Matrix4()
             .rotate_axis(40.0, np.array([1., -1.5, 0.7]))
             .translate(np.array([0.5, -0.5, 0.3])))
        rpts = np.array([M.transform(p) for p in self.pts])
        ring_t = _ring_from_pts(rpts)
        B_raw = np.array([[0., 0., -1.8], [0., 0., 1.8]])
        B = np.array([M.transform(p) for p in B_raw])
        did, pt = ring_t.pierced_by(B)
        assert did, 'tilted piercing bond should still pierce'
        return rpts, ring_t, B, pt

    # -- always-run regression tests -----------------------------------------

    def test_pierce_axial(self):
        self._case_pierce_axial()

    def test_no_pierce_same_side(self):
        self._case_no_pierce_same_side()

    def test_no_pierce_outside(self):
        self._case_no_pierce_outside()

    def test_pierce_tilted(self):
        self._case_pierce_tilted()

    # -- figure generation ---------------------------------------------------

    def test_generate_figure(self, figdir):
        if figdir is None:
            pytest.skip('pass --gen-figs to generate figures')

        B_a, pt_a = self._case_pierce_axial()
        B_b = self._case_no_pierce_same_side()
        B_c = self._case_no_pierce_outside()
        rpts_d, ring_d, B_d, pt_d = self._case_pierce_tilted()

        fig = plt.figure(figsize=(11, 9))
        fig.suptitle('Ring-pierce detection: geometry cases', fontsize=12, y=0.98)

        L = self.RADIUS * 1.6
        lims = (-L, L)
        zlims = (-2.2, 2.2)

        # (a) pierce — axial bond
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        _draw_ring(ax, self.pts)
        _draw_bond(ax, B_a[0], B_a[1])
        _mark_pierce(ax, pt_a)
        _style(ax, '(a)  Pierce — bond through centre', lims, zlims)

        # (b) no pierce — both endpoints same side
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        _draw_ring(ax, self.pts)
        _draw_bond(ax, B_b[0], B_b[1], color='darkorange')
        _style(ax, '(b)  No pierce — both endpoints same side', lims, zlims)

        # (c) no pierce — bond outside ring boundary
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        _draw_ring(ax, self.pts)
        _draw_bond(ax, B_c[0], B_c[1], color='darkorange')
        # annotate the out-of-ring intersection point for clarity
        off = self.RADIUS * 1.5
        ax.scatter(off, off, 0, color='silver', s=120, marker='x',
                   linewidth=2, zorder=8, label='plane intersection\n(outside ring)')
        _style(ax, '(c)  No pierce — intersection outside ring', lims, zlims)

        # (d) pierce — arbitrary orientation
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        _draw_ring(ax, rpts_d)
        _draw_bond(ax, B_d[0], B_d[1])
        _mark_pierce(ax, pt_d)
        _autostyle(ax, '(d)  Pierce — arbitrary orientation',
                   [rpts_d, B_d])

        # shared legend at bottom
        handles = [
            mpatches.Patch(facecolor='steelblue', alpha=0.5,
                           edgecolor='navy', label='ring'),
            plt.Line2D([0], [0], color='crimson', lw=2,
                       label='bond (pierces)'),
            plt.Line2D([0], [0], color='darkorange', lw=2,
                       label='bond (no pierce)'),
            plt.Line2D([0], [0], marker='*', color='gold', ms=12, lw=0,
                       markeredgecolor='darkorange',
                       label='intersection point'),
            plt.Line2D([0], [0], marker='x', color='silver', ms=10, lw=0,
                       markeredgewidth=2,
                       label='plane intersection\n(outside ring)'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=5,
                   fontsize=8.5, bbox_to_anchor=(0.5, 0.01))

        fig.tight_layout(rect=[0, 0.07, 1, 0.97])
        out = figdir / 'ring_pierce_cases.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'\n  Saved {out}')


# ---------------------------------------------------------------------------
# Figure 2 — linkcell neighbourhood selection (2-D top-down)
# ---------------------------------------------------------------------------

class TestRingPierceLinkcellFigure:
    """Generates ring_pierce_linkcell.png.

    Scenario: hexagon centred at (5, 5), two bond-endpoint atoms well
    separated from the ring in x.  Top-down (x-y) projection shows which
    cells are searched and why the ring is found as a candidate.
    """

    # system constants
    BOX    = np.array([10., 10., 4.])
    CUTOFF = 2.5        # → 4 × 4 × 2 cells, celldim 2.5 × 2.5 × 2
    CX, CY, CZ = 5., 5., 2.
    RADIUS = 0.8

    def _build(self):
        ring_pos = _regular_polygon(6, self.RADIUS, (self.CX, self.CY, self.CZ))
        bond_pos = np.array([[1.25, self.CY, self.CZ],
                              [8.75, self.CY, self.CZ]])
        A = _atom_df(np.vstack([ring_pos, bond_pos]), start_idx=1)
        lc = Linkcell()
        lc.create(self.CUTOFF, self.BOX)
        lc.assign(A)
        return lc, A, ring_pos, bond_pos

    # -- regression: ring found in candidate set -----------------------------

    def test_ring_in_candidate_set(self):
        lc, A, ring_pos, bond_pos = self._build()
        ring_ids = list(range(1, 7))
        bond_ids = [7, 8]

        ci = int(A.loc[A['globalIdx'] == 7, 'linkcell_idx'].iloc[0])
        cj = int(A.loc[A['globalIdx'] == 8, 'linkcell_idx'].iloc[0])
        cell_set = lc.neighbor_cell_set(ci) | lc.neighbor_cell_set(cj)
        atom_ids = lc.nearby_atom_ids(A, cell_set)

        rings = RingList([Ring(ring_ids)])
        rings.injest_coordinates(A)
        candidates = rings.filter(atom_ids)
        assert len(candidates) == 1, 'ring should be a pierce candidate'

    # -- figure generation ---------------------------------------------------

    def test_generate_figure(self, figdir):
        if figdir is None:
            pytest.skip('pass --gen-figs to generate figures')

        lc, A, ring_pos, bond_pos = self._build()

        nx_c = int(lc.ncells[0])
        ny_c = int(lc.ncells[1])
        cdx  = lc.celldim[0]
        cdy  = lc.celldim[1]

        # ---------- classify 2-D cell columns --------------------------------
        def xy_cell(x, y):
            return (int(np.floor(x / cdx)) % nx_c,
                    int(np.floor(y / cdy)) % ny_c)

        def moore_nbrs_2d(ci, cj):
            s = set()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    s.add(((ci + di) % nx_c, (cj + dj) % ny_c))
            return s

        ring_cells  = set(xy_cell(p[0], p[1]) for p in ring_pos)
        bond_cells  = set(xy_cell(p[0], p[1]) for p in bond_pos)
        bond_nbrs   = set()
        for cell in bond_cells:
            bond_nbrs |= moore_nbrs_2d(*cell)
        found_cells = ring_cells & bond_nbrs   # ring cells reachable by search

        # ---------- figure ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 8))

        COLORS = {
            'found':     '#2ecc71',   # green  — ring cell found by search
            'ring':      '#3498db',   # blue   — ring cell not yet found
            'bond':      '#e74c3c',   # red    — bond-endpoint cell
            'searched':  '#fde8e6',   # pink   — neighbour cells searched
            'default':   '#f5f5f5',   # grey   — not searched
        }

        for i in range(nx_c):
            for j in range(ny_c):
                cell = (i, j)
                if cell in found_cells:
                    fc = COLORS['found']
                elif cell in ring_cells:
                    fc = COLORS['ring']
                elif cell in bond_cells:
                    fc = COLORS['bond']
                elif cell in bond_nbrs:
                    fc = COLORS['searched']
                else:
                    fc = COLORS['default']
                rect = mpatches.FancyBboxPatch(
                    (i * cdx + 0.03, j * cdy + 0.03),
                    cdx - 0.06, cdy - 0.06,
                    boxstyle='square,pad=0',
                    facecolor=fc, edgecolor='#aaaaaa', linewidth=0.5,
                    zorder=2,
                )
                ax.add_patch(rect)
                ax.text(i * cdx + cdx / 2, j * cdy + 0.22,
                        f'({i},{j})', ha='center', va='bottom',
                        fontsize=7.5, color='#555555', zorder=4)

        # grid lines
        for i in range(nx_c + 1):
            ax.axvline(i * cdx, color='#aaaaaa', lw=0.8, zorder=1)
        for j in range(ny_c + 1):
            ax.axhline(j * cdy, color='#aaaaaa', lw=0.8, zorder=1)

        # ring polygon
        ring_patch = plt.Polygon(
            ring_pos[:, :2], closed=True,
            facecolor=COLORS['ring'], alpha=0.45,
            edgecolor='navy', lw=2, zorder=5,
        )
        ax.add_patch(ring_patch)
        ax.scatter(ring_pos[:, 0], ring_pos[:, 1],
                   color='navy', s=55, zorder=6, label='ring atoms')

        # bond endpoint atoms and the prospective bond
        ax.scatter(bond_pos[:, 0], bond_pos[:, 1],
                   color=COLORS['bond'], s=110, marker='D', zorder=7,
                   label='bond-endpoint atoms')
        ax.plot(bond_pos[:, 0], bond_pos[:, 1],
                '--', color=COLORS['bond'], lw=2, zorder=6,
                label='proposed bond (projected)')

        # labels for bond atoms
        for gidx, (x, y) in zip([7, 8], bond_pos[:, :2]):
            ax.text(x, y + 0.18, f'atom {gidx}', ha='center', fontsize=8,
                    color=COLORS['bond'], zorder=8)

        ax.set_xlim(-0.15, self.BOX[0] + 0.15)
        ax.set_ylim(-0.15, self.BOX[1] + 0.15)
        ax.set_aspect('equal')
        ax.set_xlabel('x  (nm)', fontsize=10)
        ax.set_ylabel('y  (nm)', fontsize=10)
        ax.set_title(
            'Linkcell neighbourhood search — top-down view (z projected out)\n'
            f'Box {self.BOX[0]}×{self.BOX[1]} nm,  '
            f'cutoff {self.CUTOFF} nm  →  {nx_c}×{ny_c} cells '
            f'({cdx:.1f}×{cdy:.1f} nm each)',
            fontsize=10,
        )

        # legend
        cell_handles = [
            mpatches.Patch(color=COLORS['bond'],    label='bond-endpoint cell'),
            mpatches.Patch(color=COLORS['searched'], label='searched (bond neighbours)'),
            mpatches.Patch(color=COLORS['ring'],     label='ring-atom cell'),
            mpatches.Patch(color=COLORS['found'],    label='ring cell found by search'),
        ]
        atom_handles, atom_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=atom_handles + cell_handles,
            loc='upper right', fontsize=8.5, framealpha=0.92,
        )

        # annotation arrow pointing at the found cell
        found_cell = next(iter(found_cells))
        fx = found_cell[0] * cdx + cdx / 2
        fy = found_cell[1] * cdy + cdy / 2
        ax.annotate(
            'ring found\nin search',
            xy=(fx, fy), xytext=(fx + 1.5, fy + 1.8),
            fontsize=8.5, color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
            zorder=9,
        )

        fig.tight_layout()
        out = figdir / 'ring_pierce_linkcell.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'\n  Saved {out}')
