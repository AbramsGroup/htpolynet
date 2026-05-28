"""Topology surgery primitives for postcure repair operations.

Operations the standard cure/cap machinery does not expose: severing
bonds with cascading angle/dihedral/pair cleanup, in-place atom-type
and bond-order changes, residue reassignment, and atom relocation.
Atom deletions still go through ``TopoCoord.delete_atoms`` (which
reindexes everything); callers should batch deletions to the end of a
surgery pass so intermediate index references remain valid.
"""
import logging
import numpy as np
import pandas as pd

from ..geometry.bondlist import Bondlist

logger = logging.getLogger(__name__)


def _canonical_pair_set(pairs):
    """Convert an iterable of (i, j) pairs into a set of (min, max) tuples."""
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs}


def _canonical_pair_mask(df, col_a, col_b, pair_set):
    """Vectorized mask: True where (df[col_a], df[col_b]) (in either order)
    appears in pair_set."""
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)
    ai = df[col_a].to_numpy()
    aj = df[col_b].to_numpy()
    lo = np.minimum(ai, aj)
    hi = np.maximum(ai, aj)
    canonical = list(zip(lo.tolist(), hi.tolist()))
    return pd.Series([p in pair_set for p in canonical], index=df.index)


def delete_bonds(TC, pairs):
    """Remove bonds and cascade-delete dependent angles/dihedrals/pairs.

    Atom indices are not changed.  Any [angles] whose i-j or j-k sequence
    is a deleted bond are removed; any [dihedrals] whose i-j, j-k, or k-l
    sequence is a deleted bond are removed; stale [pairs] entries left
    over are pruned via Topology.prune_stale_14_pairs.

    Args:
        TC: TopoCoord
        pairs (Iterable[Tuple[int,int]]): bond endpoints (1-based atom indices)

    Returns:
        dict: counts of removed entries per section
    """
    pair_set = _canonical_pair_set(pairs)
    if not pair_set:
        return {'bonds': 0, 'angles': 0, 'dihedrals': 0}
    T = TC.Topology
    counts = {}

    bonds = T.D['bonds']
    bmask = _canonical_pair_mask(bonds, 'ai', 'aj', pair_set)
    counts['bonds'] = int(bmask.sum())
    T.D['bonds'] = bonds[~bmask].reset_index(drop=True)
    T.bondlist = Bondlist.fromDataFrame(T.D['bonds'])

    if 'mol2_bonds' in T.D and not T.D['mol2_bonds'].empty:
        m = T.D['mol2_bonds']
        mmask = _canonical_pair_mask(m, 'ai', 'aj', pair_set)
        T.D['mol2_bonds'] = m[~mmask].reset_index(drop=True)
        T.D['mol2_bonds']['bondIdx'] = list(range(1, T.D['mol2_bonds'].shape[0] + 1))

    angles = T.D['angles']
    amask = _canonical_pair_mask(angles, 'ai', 'aj', pair_set) \
          | _canonical_pair_mask(angles, 'aj', 'ak', pair_set)
    counts['angles'] = int(amask.sum())
    T.D['angles'] = angles[~amask].reset_index(drop=True)

    dihedrals = T.D['dihedrals']
    dmask = _canonical_pair_mask(dihedrals, 'ai', 'aj', pair_set) \
          | _canonical_pair_mask(dihedrals, 'aj', 'ak', pair_set) \
          | _canonical_pair_mask(dihedrals, 'ak', 'al', pair_set)
    counts['dihedrals'] = int(dmask.sum())
    T.D['dihedrals'] = dihedrals[~dmask].reset_index(drop=True)

    dropped_pairs = T.prune_stale_14_pairs()
    counts['pairs'] = int(dropped_pairs or 0)

    logger.debug(
        f'delete_bonds: removed {counts["bonds"]} bonds, '
        f'{counts["angles"]} angles, {counts["dihedrals"]} dihedrals, '
        f'{counts["pairs"]} stale 14-pairs'
    )
    return counts


def set_bond_order(TC, ai, aj, order):
    """Set the bond-order (funct) for an existing bond.  Caller is responsible
    for refreshing parameters via reset_override_from_type if atom types
    changed."""
    bonds = TC.Topology.D['bonds']
    lo, hi = min(int(ai), int(aj)), max(int(ai), int(aj))
    a, b = bonds['ai'].to_numpy(), bonds['aj'].to_numpy()
    lo_col = np.minimum(a, b)
    hi_col = np.maximum(a, b)
    mask = (lo_col == lo) & (hi_col == hi)
    n = int(mask.sum())
    if n == 0:
        logger.debug(f'set_bond_order: no bond between {ai} and {aj}')
        return False
    bonds.loc[mask, 'funct'] = int(order)
    return True


def set_atom_attributes(TC, idx, **attrs):
    """Update one atom's coord-side (gro) and topology-side (top) attributes.

    Mirrors the column-name mismatch between Coordinates.A (atomName, resNum,
    resName, posX/Y/Z) and Topology.D['atoms'] (atom, resnr, residue) so a
    single call updates both sides.

    Recognized keys: atomName, resName, resNum, type, charge, posX, posY, posZ.
    """
    idx = int(idx)
    A = TC.Coordinates.A
    atdf = TC.Topology.D['atoms']
    a_mask = A['globalIdx'] == idx
    t_mask = atdf['nr'] == idx
    for k, v in attrs.items():
        if k == 'atomName':
            A.loc[a_mask, 'atomName'] = v
            atdf.loc[t_mask, 'atom'] = v
        elif k == 'resName':
            A.loc[a_mask, 'resName'] = v
            atdf.loc[t_mask, 'residue'] = v
        elif k == 'resNum':
            A.loc[a_mask, 'resNum'] = int(v)
            atdf.loc[t_mask, 'resnr'] = int(v)
        elif k == 'type':
            atdf.loc[t_mask, 'type'] = v
        elif k == 'charge':
            atdf.loc[t_mask, 'charge'] = float(v)
        elif k in ('posX', 'posY', 'posZ'):
            A.loc[a_mask, k] = float(v)
        else:
            raise ValueError(f'set_atom_attributes: unknown attribute {k!r}')


def reassign_residue(TC, atom_idxs, new_resname, new_resnum):
    """Move a set of atoms into a (possibly new) residue."""
    for idx in atom_idxs:
        set_atom_attributes(TC, idx, resName=new_resname, resNum=int(new_resnum))


def next_free_resnum(TC):
    """Return a resNum guaranteed to not collide with any existing residue."""
    return int(TC.Coordinates.A['resNum'].max()) + 1


def refresh_bond_params(TC, pairs):
    """For each bond, clear cached c0/c1 so they re-resolve from [bondtypes]
    against the current atom types.  Call after set_atom_attributes(type=...)
    on either bond endpoint."""
    for ai, aj in pairs:
        try:
            TC.Topology.reset_override_from_type('bonds', 'bondtypes',
                                                 inst_idx=(int(ai), int(aj)))
        except Exception as e:
            logger.debug(f'refresh_bond_params({ai},{aj}) failed: {e}')


def add_bonds_with_template(TC, pairs, moldict, product_name, chain_manager=None):
    """Add new bonds and splice angles/dihedrals/pairs/types/charges from a
    pre-parameterized linked-product template.

    Wraps TopoCoord.make_bonds + TopoCoord.map_from_templates with the
    repair-side convention that the reactive atoms have already lost their
    sacrificial H's (so explicit_sacH is empty for every new bond).

    Args:
        TC: TopoCoord
        pairs: list of (ai, aj, order) tuples
        moldict: MoleculeDict
        product_name: the linked-product Molecule name in moldict whose
            template parameters should be spliced into the system around
            each new bond.
        chain_manager: optional ChainManager owned by the caller.

    Returns:
        list: sacrificial-H indices that make_bonds would have deleted
            (empty when explicit_sacH={[]} for all pairs).
    """
    if not pairs:
        return []
    explicit_sacH = {i: [] for i in range(len(pairs))}
    idx_to_delete = TC.make_bonds(pairs, explicit_sacH=explicit_sacH,
                                  chain_manager=chain_manager)
    bdf = pd.DataFrame([{
        'ai': int(ai),
        'aj': int(aj),
        'order': int(order),
        'reactantName': product_name,
    } for ai, aj, order in pairs])
    TC.map_from_templates(bdf, moldict, chain_manager=chain_manager)
    # map_from_templates concats new rows that came from .map(temp2inst) onto
    # the existing angles/dihedrals/pairs frames.  Even with NaN-bearing rows
    # filtered out before the concat, pandas can upcast int atom-index columns
    # to float64 when intermediate rows held NaN. delete_atoms later asserts
    # int dtype on these columns, so coerce back here.
    _fix_atom_index_dtypes(TC)
    return idx_to_delete


def _fix_atom_index_dtypes(TC):
    """Coerce atom-index columns in bonds/angles/dihedrals/pairs back to int
    after any operation that may have upcast them via NaN-tainted concat."""
    for section, cols in (('bonds', ('ai', 'aj')),
                          ('pairs', ('ai', 'aj')),
                          ('angles', ('ai', 'aj', 'ak')),
                          ('dihedrals', ('ai', 'aj', 'ak', 'al'))):
        d = TC.Topology.D.get(section)
        if d is None or d.empty:
            continue
        for c in cols:
            if c in d.columns and d[c].dtype != int:
                if d[c].isna().any():
                    logger.warning(
                        f'_fix_atom_index_dtypes: NaN in {section}.{c}; '
                        f'dropping {int(d[c].isna().sum())} rows'
                    )
                    d = d.dropna(subset=[c]).reset_index(drop=True)
                    TC.Topology.D[section] = d
                d[c] = d[c].astype(int)


def find_atoms(TC, **filters):
    """Return globalIdx list of atoms matching the given (column == value)
    filters.  Convenience wrapper around the Coordinates dataframe."""
    A = TC.Coordinates.A
    mask = pd.Series([True] * len(A), index=A.index)
    for col, val in filters.items():
        mask &= (A[col] == val)
    return A.loc[mask, 'globalIdx'].astype(int).tolist()


def get_attr(TC, idx, col):
    """Read one column for one atom from the coord-side dataframe."""
    return TC.Coordinates.A.loc[TC.Coordinates.A['globalIdx'] == int(idx),
                                col].iloc[0]


def positions(TC, idxs):
    """Return an (n, 3) ndarray of positions for the given atom indices."""
    A = TC.Coordinates.A
    sub = A.loc[A['globalIdx'].isin([int(i) for i in idxs]),
                ['globalIdx', 'posX', 'posY', 'posZ']]
    # preserve caller ordering
    pos = {int(r.globalIdx): (r.posX, r.posY, r.posZ) for r in sub.itertuples()}
    return np.array([pos[int(i)] for i in idxs])
