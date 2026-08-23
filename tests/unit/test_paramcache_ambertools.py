"""End-to-end checks that the provenance record describes what really happened.

tests/unit/test_paramcache.py pins the comparison logic without running
AmberTools -- it compares records rather than producing them.  What it cannot
show is that the record matches reality, or that the directives it records make
any difference to the numbers.  This module runs antechamber twice on the same
molecule, once per charge method, and pins both.

Skips when the AmberTools chain is absent, as the other integration modules do.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import os
import shutil

from types import SimpleNamespace

import pytest

import htpolynet.core.projectfilesystem as pfs

from htpolynet.core import paramcache
from htpolynet.core.coordinates import Coordinates
from htpolynet.core.molecule import Molecule
from htpolynet.core.runtime import Runtime

_REQUIRED_TOOLS = ('antechamber', 'parmchk2', 'tleap', 'gmx')
_MISSING = [t for t in _REQUIRED_TOOLS if shutil.which(t) is None]
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f'external tools not on PATH: {", ".join(_MISSING)}',
)


def _parameterize(tmpdir, charge_method):
    """Parameterizes the shipped STY monomer and returns its charges and record.

    Returns:
        tuple: (pandas.Series of charge by atom name, provenance record dict)
    """
    orig = os.getcwd()
    pfs.pfs_setup(root=str(tmpdir), projdir='proj', mock=False)
    try:
        M = Molecule.New('STY', None)
        M.set_sequence_from_moldict({'STY': M})
        M.generate(ambertools={'charge_method': charge_method},
                   gaff={'minimize_molecules': False})
        charges = Coordinates.read_mol2('STY.mol2').A.set_index('atomName')['charge']
        return charges, paramcache.read_key('STY')
    finally:
        os.chdir(orig)


@pytest.fixture(scope='module')
def gas(tmp_path_factory):
    return _parameterize(tmp_path_factory.mktemp('sty_gas'), 'gas')


@pytest.fixture(scope='module')
def bcc(tmp_path_factory):
    return _parameterize(tmp_path_factory.mktemp('sty_bcc'), 'bcc')


class TestRecordDescribesWhatRan:

    def test_gas_run_is_recorded_as_gas(self, gas):
        _, record = gas
        assert record is not None, 'a parameterization must leave a record'
        assert record['charge_method'] == 'gas'

    def test_bcc_run_is_recorded_as_bcc(self, bcc):
        _, record = bcc
        assert record is not None
        assert record['charge_method'] == 'bcc'

    def test_record_carries_the_net_charge_that_was_used(self, gas):
        _, record = gas
        assert record['net_charge'] == 0
        assert record['atom_type'] == 'gaff'


class TestTheDirectiveActuallyMatters:
    """Without this, guarding the cache on charge_method would be ceremony."""

    def test_the_two_methods_give_different_charges(self, gas, bcc):
        gas_q, _ = gas
        bcc_q, _ = bcc
        assert set(gas_q.index) == set(bcc_q.index), 'same molecule, same atoms'
        assert not gas_q.equals(bcc_q[gas_q.index]), \
            'gas and bcc must not produce identical charges'

    def test_the_difference_is_large_on_at_least_one_atom(self, gas, bcc):
        # The reported failure was 5.5x on a triazine ring carbon.  Styrene is
        # a milder case, so this pins only that the gap is far outside anything
        # attributable to rounding in the mol2 charge field.
        gas_q, _ = gas
        bcc_q, _ = bcc
        assert (bcc_q[gas_q.index] - gas_q).abs().max() > 0.05

    def test_both_are_neutral_overall(self, gas, bcc):
        # Whatever the method, -nc 0 must still be honored; a charge model that
        # silently changed the net charge would be a different bug.
        for charges, _ in (gas, bcc):
            assert abs(charges.sum()) < 1e-2


# ---------------------------------------------------------------------------
# The reuse path: a run meeting a library entry built the other way.
#
# The tests above show the record describes what ran and that the directive
# matters.  Neither shows the thing that actually broke -- a build finding a
# gas entry under the name it wants and having to refuse it.  That needs a
# library with something in it, so it gets its own fixture.
# ---------------------------------------------------------------------------

@pytest.fixture
def throwaway_library(tmp_path, monkeypatch):
    """A user cache in tmp_path, bound explicitly rather than by environment.

    pfs._USER_CACHE_ is a module global initialized on first use, so setting
    HTPOLYNET_CACHE would be ignored if anything earlier in the session had
    already bound it.  Patching the object itself is what guarantees no path
    through these tests can reach the real ~/.htpolynet -- which holds work
    these tests must not read from and must never write to.
    """
    cache = tmp_path / 'cache'
    monkeypatch.setattr(pfs, '_USER_CACHE_', pfs.UserCache(path=str(cache)))
    orig = os.getcwd()
    yield tmp_path, cache / pfs.Dirs.molecules_parameterized
    os.chdir(orig)


def _build(root, charge_method, force_parameterization=False):
    """Runs one molecule through Runtime's generate-or-reuse decision.

    Returns:
        tuple: (origin string, charges by atom name)
    """
    root.mkdir(parents=True, exist_ok=True)
    pfs.pfs_setup(root=str(root), projdir='proj', mock=False)
    pfs.go_to(pfs.Dirs.molecules_parameterized)
    M = Molecule.New('STY', None)
    M.set_sequence_from_moldict({'STY': M})
    M.origin = 'unparameterized'
    M.zrecs = [{'resid': 1, 'atom': 'C1', 'z': 1}, {'resid': 1, 'atom': 'C2', 'z': 1}]
    # A real Runtime without __init__, so the methods under test are the real
    # ones while the configuration is exactly what this build asks for.
    R = Runtime.__new__(Runtime)
    R.cfg = SimpleNamespace(ambertools={'charge_method': charge_method},
                            gaff={'minimize_molecules': False})
    R.molecules = {'STY': M}
    R.unverified_parameterizations = []
    R._generate_molecule(M, force_parameterization=force_parameterization,
                         force_checkin=False)
    charges = Coordinates.read_mol2('STY.mol2').A.set_index('atomName')['charge']
    return M.origin, charges


class TestStaleCacheIsRefused:

    def test_a_matching_cache_is_reused(self, throwaway_library):
        # The control.  Without this, "always re-parameterizes" would pass
        # every other test in this class.
        root, _ = throwaway_library
        first, gas_q = _build(root / 'a', 'gas')
        assert first == 'newly parameterized'
        second, again = _build(root / 'b', 'gas')
        assert second == 'previously parameterized', 'a matching entry must still be reused'
        assert again.equals(gas_q)

    def test_a_gas_entry_does_not_satisfy_a_bcc_run(self, throwaway_library):
        # The reported bug, end to end: the library holds gas under the name
        # the config wants, and the config asks for bcc.
        root, _ = throwaway_library
        _, gas_q = _build(root / 'a', 'gas')
        origin, bcc_q = _build(root / 'b', 'bcc')
        assert origin == 'newly parameterized', 'the gas entry must be refused, not reused'
        assert not bcc_q.equals(gas_q), 'the build must carry bcc charges, not the cached gas ones'

    def test_the_refused_entry_is_not_overwritten_without_force_checkin(self, throwaway_library):
        # The library is someone's accumulated work.  Refusing an entry must
        # not replace it; that needs --force-checkin.
        root, library = throwaway_library
        _build(root / 'a', 'gas')
        cached_gas = Coordinates.read_mol2(str(library / 'STY.mol2')).A['charge'].copy()
        cached_record = paramcache.read_key(str(library / 'STY'))
        _build(root / 'b', 'bcc')
        assert Coordinates.read_mol2(str(library / 'STY.mol2')).A['charge'].equals(cached_gas)
        assert paramcache.read_key(str(library / 'STY')) == cached_record
        assert cached_record['charge_method'] == 'gas'

    def test_the_run_says_which_directive_differed(self, throwaway_library, caplog):
        root, _ = throwaway_library
        _build(root / 'a', 'gas')
        with caplog.at_level('INFO'):
            _build(root / 'b', 'bcc')
        assert "'gas'" in caplog.text and "'bcc'" in caplog.text
        assert 'charge method' in caplog.text
