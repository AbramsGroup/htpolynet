"""Postcure topology-repair operations.

Repair drivers transform a cured system in ways the monotonic
cure/cap reaction machinery cannot: severing bonds, deleting atoms in
bulk, transferring atoms between residues, and re-templating affected
linkages.  Each driver lives in its own module and is dispatched from
``run_repair`` based on the ``type`` field of a postcure_repair entry.
"""
import logging

logger = logging.getLogger(__name__)


def run_repair(TC, moldict, repair_specs, reactions):
    """Dispatch postcure repair operations.

    Args:
        TC: TopoCoord for the cured system (modified in place).
        moldict: MoleculeDict containing all parameterized templates,
            including any repair-stage linked-product templates.
        repair_specs: list of repair-spec dicts from the config
            (Configuration.postcure_repair).
        reactions: ReactionList including repair-stage reactions used as
            parameter templates by the drivers.

    Returns:
        tuple: (int, list) -- the number of repairs performed across all
            drivers, and one statistics dict per driver that reported any
            (see :func:`htpolynet.repair.cyanate_cap._completion_stats`).
    """
    if not repair_specs:
        return 0, []
    total = 0
    stats = []
    for spec in repair_specs:
        rtype = spec.get('type')
        if rtype == 'triazine_to_cyanate_cap':
            from .cyanate_cap import triazine_to_cyanate_cap
            result = triazine_to_cyanate_cap(TC, moldict, spec, reactions)
            total += result['n_dismantled']
            stats.append(result)
        else:
            logger.warning(f'Unknown postcure_repair type "{rtype}"; skipping')
    return total, stats
