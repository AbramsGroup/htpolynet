"""Triazine-to-cyanate-cap postcure repair.

Dismantles every incomplete triazine crosslinker (one with fewer than the
configured ``full_bond_count`` BPA-O bonds) into three independent -C#N
fragments, each then attached either in place (to the BPA-O it was already
bonded to during cure) or to a free, unreacted BPA-OH within
``cap_search_radius``.  Atom conservation is exact: across the whole
system, the count of fragments freed from incomplete rings equals the
count of unreacted BPA-OHs, so every free fragment finds a home.

The surgery is staged so that atom-index references stay valid until the
final batched deletion of sacrificial H atoms; intermediate phases edit
bonds, atom attributes, and residue tags only.
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from . import topology_surgery as ts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------


def _find_crosslinker_residues(TC, residue_name):
    """Return {resNum: [globalIdx, ...]} for every residue with this name."""
    A = TC.Coordinates.A
    sub = A[A['resName'] == residue_name][['globalIdx', 'resNum', 'atomName']]
    out = {}
    for rn, g in sub.groupby('resNum'):
        out[int(rn)] = {row.atomName: int(row.globalIdx) for row in g.itertuples()}
    return out


def _bonded_bridge_o(TC, c_idx, bridge_residue, bridge_oxygens):
    """If atom c_idx is bonded to a (bridge_residue).(one of bridge_oxygens),
    return the bridge-O globalIdx; else return None."""
    A = TC.Coordinates.A
    for nbr in TC.Topology.bondlist.partners_of(int(c_idx)):
        row = A.loc[A['globalIdx'] == int(nbr)].iloc[0]
        if row['resName'] == bridge_residue and row['atomName'] in bridge_oxygens:
            return int(nbr)
    return None


def _ring_matching(ring_carbons, ring_nitrogens, bondlist):
    """Choose a perfect C-N matching in the 6-cycle ring.

    For a 1,3,5-triazine ring with carbons C1, C2, C3 and nitrogens
    N1, N2, N3 (in ring-traversal order ``C1-N1-C2-N2-C3-N3-C1``), both
    possible matchings are valid; we pick the one that pairs each C with
    its predecessor N in traversal order, i.e. ``Ci`` with the N that
    comes immediately before it.

    Args:
        ring_carbons (list[int]): C atom indices in ring traversal order.
        ring_nitrogens (list[int]): N atom indices in ring traversal order,
            interleaved with the carbons (N1 is between C1 and C2, etc.).
        bondlist: TC.Topology.bondlist for sanity-checking adjacency.

    Returns:
        list[tuple[int, int]]: list of (c_idx, n_idx) matched pairs.
    """
    matching = []
    for i, c in enumerate(ring_carbons):
        n = ring_nitrogens[(i - 1) % len(ring_nitrogens)]
        assert bondlist.are_bonded(c, n), (
            f'ring_matching: expected C{i+1}={c} bonded to N={n} '
            f'but they are not adjacent'
        )
        matching.append((int(c), int(n)))
    return matching


def _severed_ring_bonds(ring_carbons, ring_nitrogens, matching):
    """The 3 ring N-C bonds that are NOT in the matching."""
    kept = {(min(c, n), max(c, n)) for c, n in matching}
    severed = []
    # Ring edges in traversal: C1-N1, N1-C2, C2-N2, N2-C3, C3-N3, N3-C1
    n_ring = len(ring_carbons)
    for i in range(n_ring):
        c = ring_carbons[i]
        # the two N's adjacent to this C
        n_prev = ring_nitrogens[(i - 1) % n_ring]
        n_next = ring_nitrogens[i]
        for n in (n_prev, n_next):
            pair = (min(c, n), max(c, n))
            if pair not in kept and pair not in {(a, b) for a, b in severed}:
                severed.append((c, n))
    return severed


# ---------------------------------------------------------------------------
# Geometry placement
# ---------------------------------------------------------------------------


def _place_cyn_along(bpa_o_xyz, bpa_o_h_xyz, oc_len=0.136, cn_len=0.116):
    """Compute initial placement for a transferred -C#N group along the
    direction from BPA-O toward the (about-to-be-deleted) H, so the new
    cap sticks out along the original O-H bond.

    Lengths in nm: standard -O-C ester ~1.36 A, -C#N ~1.16 A.
    Returns (C_xyz, N_xyz).
    """
    o = np.asarray(bpa_o_xyz, dtype=float)
    h = np.asarray(bpa_o_h_xyz, dtype=float)
    direction = h - o
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    u = direction / norm
    c_xyz = o + u * oc_len
    n_xyz = c_xyz + u * cn_len
    return c_xyz, n_xyz


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def triazine_to_cyanate_cap(TC, moldict, spec, reactions):
    """Execute the triazine-to-cyanate-cap repair.

    Args:
        TC: TopoCoord (modified in place).
        moldict: MoleculeDict including the linked-product template that
            describes a fully repaired BPA-O-C#N cap (looked up by the
            ``cap_template`` name in spec, with symmetry siblings auto-tried).
        spec: dict from postcure_repair config (see example 6 YAML).
        reactions: ReactionList (currently unused; reserved for cross-checks).

    Returns:
        int: number of incomplete triazines dismantled.
    """
    cl = spec['crosslinker']
    br = spec['bridge']
    cap_residue_name = spec['cap_residue']
    cap_template = spec['cap_template']
    full_bond_count = int(cl.get('full_bond_count', 3))
    ring_c_names = cl['ring_carbon_atoms']      # e.g. ['C1', 'C2', 'C3']
    ring_n_names = cl['ring_nitrogen_atoms']    # e.g. ['N1', 'N2', 'N3']
    bridge_residue = br['residue']
    bridge_o_names = br['reactive_oxygen_atoms']  # e.g. ['O1', 'O2']
    search_radius = float(spec.get('cap_search_radius', 0.6))

    bondlist = TC.Topology.bondlist

    # ---- Phase 1: find incomplete crosslinker residues and plan caps ----
    cl_residues = _find_crosslinker_residues(TC, cl['residue'])
    if not cl_residues:
        logger.info(f'triazine_to_cyanate_cap: no {cl["residue"]} residues found; nothing to do')
        return 0

    incomplete_plans = []     # list of dicts, one per incomplete triazine
    h_to_delete = set()       # global H indices, batched for final deletion

    for resnum, atoms_by_name in cl_residues.items():
        ring_c = [atoms_by_name[n] for n in ring_c_names]
        ring_n = [atoms_by_name[n] for n in ring_n_names]
        bonded_o = [_bonded_bridge_o(TC, c, bridge_residue, bridge_o_names) for c in ring_c]
        k = sum(1 for o in bonded_o if o is not None)
        if k >= full_bond_count:
            continue
        # plan dismantle
        matching = _ring_matching(ring_c, ring_n, bondlist)
        severed = _severed_ring_bonds(ring_c, ring_n, matching)
        # plan H deletions: each dangling ring C has one C-H to remove
        for c_idx, o_partner in zip(ring_c, bonded_o):
            if o_partner is None:
                for nbr in bondlist.partners_of(c_idx):
                    name = ts.get_attr(TC, nbr, 'atomName')
                    if name.upper().startswith('H'):
                        h_to_delete.add(int(nbr))
        # one cap per (c, n) pair: tag in-place vs free
        caps = []
        c_to_o = dict(zip(ring_c, bonded_o))
        for c, n in matching:
            caps.append({
                'c': c,
                'n': n,
                'bonded_o': c_to_o[c],   # None for free caps
                'taz_resnum': resnum,
            })
        incomplete_plans.append({'resnum': resnum, 'caps': caps, 'severed': severed})

    if not incomplete_plans:
        logger.info(
            f'triazine_to_cyanate_cap: all {len(cl_residues)} {cl["residue"]} '
            f'residues are fully bonded; nothing to do'
        )
        return 0

    total_caps = sum(len(p['caps']) for p in incomplete_plans)
    free_caps_planned = sum(1 for p in incomplete_plans for c in p['caps'] if c['bonded_o'] is None)
    logger.info(
        f'triazine_to_cyanate_cap: {len(incomplete_plans)} incomplete '
        f'{cl["residue"]} residues identified ({total_caps} caps total, '
        f'{free_caps_planned} free fragments to donate)'
    )

    # ---- Phase 2: greedy-match free caps to unreacted bridge-O's ----
    free_caps = [c for p in incomplete_plans for c in p['caps'] if c['bonded_o'] is None]
    unreacted_o = _find_unreacted_bridge_oxygens(TC, bridge_residue, bridge_o_names)
    if len(unreacted_o) != len(free_caps):
        logger.warning(
            f'triazine_to_cyanate_cap: atom-conservation mismatch — '
            f'{len(free_caps)} free fragments vs {len(unreacted_o)} unreacted '
            f'{bridge_residue}-O sites.  Excess fragments will be discarded '
            f'(their atoms will be deleted).'
        )
    donations = _greedy_match(TC, free_caps, unreacted_o, search_radius)

    # ---- Phase 3: execute surgery (atom indices stable through phase 5) ----
    # 3a. delete all severed ring N-C bonds (cascades angles/dihedrals)
    all_severed = [b for p in incomplete_plans for b in p['severed']]
    ts.delete_bonds(TC, all_severed)

    # 3b. for in-place caps: also delete the pre-existing BPA-O-C bond so
    # add_bonds_with_template can re-form it with CYN-context parameters.
    in_place_bonds_to_redo = []
    for p in incomplete_plans:
        for cap in p['caps']:
            if cap['bonded_o'] is not None:
                in_place_bonds_to_redo.append((cap['bonded_o'], cap['c']))
    ts.delete_bonds(TC, in_place_bonds_to_redo)

    # 3c. re-tag each (c, n) pair into its own CYN residue
    new_resnum = ts.next_free_resnum(TC)
    cap_records = []  # list of (cap_dict, bridge_o_idx, new_resnum)
    for p in incomplete_plans:
        for cap in p['caps']:
            bridge_o = donations.get(id(cap))  # may be None for the in-place case
            if cap['bonded_o'] is not None:
                target_o = cap['bonded_o']
            else:
                target_o = bridge_o  # may still be None if no donation found
            if target_o is None:
                # no home for this fragment — schedule its atoms for deletion
                h_to_delete.add(int(cap['c']))
                h_to_delete.add(int(cap['n']))
                continue
            ts.reassign_residue(TC, [cap['c'], cap['n']], cap_residue_name, new_resnum)
            ts.set_atom_attributes(TC, cap['c'], atomName='C1')
            ts.set_atom_attributes(TC, cap['n'], atomName='N1')
            cap_records.append({'c': cap['c'], 'n': cap['n'], 'o': target_o, 'resnum': new_resnum})
            new_resnum += 1

    # 3d. for free-cap donations: relocate the CYN atoms next to their target O
    for rec in cap_records:
        cap_in_plans = next(
            (c for p in incomplete_plans for c in p['caps']
             if c['c'] == rec['c'] and c['n'] == rec['n']),
            None,
        )
        if cap_in_plans is None or cap_in_plans['bonded_o'] is not None:
            continue
        o_idx = rec['o']
        o_xyz = ts.positions(TC, [o_idx])[0]
        # use the (about-to-be-deleted) H on the O as the direction reference
        o_h_idx = _find_bonded_hydrogen(TC, o_idx)
        if o_h_idx is None:
            # no H to use as reference (shouldn't happen for a still-reactive O,
            # but if it does, just push the cap along +x by the rest length)
            ref_xyz = o_xyz + np.array([1.0, 0.0, 0.0])
        else:
            ref_xyz = ts.positions(TC, [o_h_idx])[0]
            h_to_delete.add(int(o_h_idx))
        c_xyz, n_xyz = _place_cyn_along(o_xyz, ref_xyz)
        ts.set_atom_attributes(TC, rec['c'],
                               posX=float(c_xyz[0]), posY=float(c_xyz[1]), posZ=float(c_xyz[2]))
        ts.set_atom_attributes(TC, rec['n'],
                               posX=float(n_xyz[0]), posY=float(n_xyz[1]), posZ=float(n_xyz[2]))

    # ---- Phase 4: splice template params (forms BPA-O-C bond with full
    # angle/dihedral/pair entries from the cap_template molecule) ----
    pairs_to_add = [(int(rec['o']), int(rec['c']), 1) for rec in cap_records]
    if pairs_to_add:
        ts.add_bonds_with_template(TC, pairs_to_add, moldict, cap_template)

    # Refresh the C-N triple-bond parameters: map_from_templates updated the
    # CYN atom types (CA -> c1, NB -> n1) but only re-resolved the bond it
    # was actively mapping (BPA.O - CYN.C).  The C-N bond carries stale
    # aromatic CA-NB override parameters until we reset it.
    ts.refresh_bond_params(TC, [(rec['c'], rec['n']) for rec in cap_records])

    # zero out reactivity on every atom touched by the repair
    for rec in cap_records:
        for idx in (rec['o'], rec['c'], rec['n']):
            TC.set_gro_attribute_by_attributes('z', 0, {'globalIdx': int(idx)})
            TC.set_gro_attribute_by_attributes('nreactions', 1, {'globalIdx': int(idx)})

    # ---- Phase 5: batched H + orphan-atom deletion (reindexes) ----
    if h_to_delete:
        logger.debug(f'triazine_to_cyanate_cap: deleting {len(h_to_delete)} sacrificial/orphan atoms')
        # Collect the heavy-atom neighbors of every deleted atom *before*
        # delete_atoms reindexes; we'll redistribute the deleted atoms'
        # missing charge back across these neighbors so the system stays
        # net-neutral.  Otherwise the lost (typically positive) H charges
        # leave the system with a several-electron net charge that gmx
        # refuses to run with Ewald electrostatics.
        affected_neighbors = set()
        for d_idx in h_to_delete:
            for nbr in TC.Topology.bondlist.partners_of(int(d_idx)):
                if int(nbr) not in h_to_delete:
                    affected_neighbors.add(int(nbr))
        idx_mapper = TC.delete_atoms(sorted(h_to_delete))
        remapped = [idx_mapper[a] for a in affected_neighbors if a in idx_mapper]
        residual = TC.Topology.total_charge()
        if remapped and abs(residual) > 1e-6:
            logger.info(
                f'triazine_to_cyanate_cap: redistributing residual charge '
                f'{residual:+.4f} across {len(remapped)} repaired-residue neighbours'
            )
            TC.Topology.adjust_charges(
                atoms=remapped,
                desired_charge=0.0,
                msg='triazine_to_cyanate_cap post-deletion rebalance',
            )

    return len(incomplete_plans)


# ---------------------------------------------------------------------------
# Helpers for matching free caps to unreacted bridge-O's
# ---------------------------------------------------------------------------


def _find_unreacted_bridge_oxygens(TC, bridge_residue, bridge_o_names):
    """Return list of (globalIdx, np.ndarray xyz) for every bridge-O atom
    that still carries a phenolic H — i.e. did not bond to any
    crosslinker during cure."""
    A = TC.Coordinates.A
    sub = A[(A['resName'] == bridge_residue) & (A['atomName'].isin(bridge_o_names))]
    unreacted = []
    for r in sub.itertuples():
        if _find_bonded_hydrogen(TC, int(r.globalIdx)) is not None:
            unreacted.append((int(r.globalIdx),
                              np.array([r.posX, r.posY, r.posZ])))
    return unreacted


def _find_bonded_hydrogen(TC, idx):
    """Return globalIdx of an H bonded to ``idx``, or None."""
    for nbr in TC.Topology.bondlist.partners_of(int(idx)):
        name = ts.get_attr(TC, nbr, 'atomName')
        if str(name).upper().startswith('H'):
            return int(nbr)
    return None


def _greedy_match(TC, free_caps, unreacted_o, search_radius):
    """Match each free cap to its nearest available unreacted bridge-O.

    Iterates free caps in order, each grabs the closest still-unmatched
    bridge-O within ``search_radius``; on a miss, the radius is expanded
    by a factor of 2 each retry up to 10x; final fallback is the globally
    nearest still-unmatched bridge-O.

    Returns dict keyed by id(cap dict) -> bridge-O globalIdx.
    """
    if not free_caps or not unreacted_o:
        return {}
    box = TC.Coordinates.box.diagonal()
    cap_positions = ts.positions(TC, [c['c'] for c in free_caps])
    o_positions = np.array([xyz for _, xyz in unreacted_o])
    o_indices = [i for i, _ in unreacted_o]
    o_available = np.ones(len(o_indices), dtype=bool)

    out = {}
    for cap, c_xyz in zip(free_caps, cap_positions):
        d = _min_image_dist(c_xyz, o_positions, box)
        d = np.where(o_available, d, np.inf)
        # progressive radius expansion
        chosen = None
        radius = search_radius
        for _ in range(10):
            idx = int(np.argmin(d))
            if np.isfinite(d[idx]) and d[idx] <= radius:
                chosen = idx
                break
            radius *= 2.0
        if chosen is None:
            # global fallback
            chosen = int(np.argmin(d))
            if not np.isfinite(d[chosen]):
                logger.warning(
                    f'_greedy_match: cap c={cap["c"]} has no available bridge-O '
                    f'remaining; will be discarded'
                )
                continue
        o_available[chosen] = False
        out[id(cap)] = int(o_indices[chosen])
    return out


def _min_image_dist(p, qs, box):
    """Minimum-image distance from p to each row of qs in an orthorhombic box."""
    delta = qs - p
    delta -= box * np.round(delta / box)
    return np.linalg.norm(delta, axis=1)
