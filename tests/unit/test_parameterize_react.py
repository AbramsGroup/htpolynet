"""Integration tests for molecule parameterisation and inter-monomer reaction.

These tests exercise the full AmberTools + GROMACS pipeline:
  1. Parameterise the STY monomer (antechamber → tleap → parmed → gmx).
  2. React two STY monomers to form the STY1_1 dimer (merge, bond, re-parameterise).

All external tools (antechamber, tleap, parmchk2, gmx) are expected to be on
PATH — conftest.py prepends the active conda environment's bin dir automatically.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import os
import pytest

from htpolynet.core.molecule import Molecule
from htpolynet.cure.reaction import Reaction, reaction_stage
import htpolynet.core.projectfilesystem as pfs


# ---------------------------------------------------------------------------
# Reaction spec: STY + STY → STY1_1  (from pSTY.yaml)
# ---------------------------------------------------------------------------
_STY1_1_SPEC = {
    'name':      'sty1_1',
    'stage':     'cure',
    'reactants': {1: 'STY', 2: 'STY'},
    'product':   'STY1_1',
    'probability': 1.0,
    'atoms': {
        'A': {'reactant': 1, 'resid': 1, 'atom': 'C1', 'z': 1},
        'B': {'reactant': 2, 'resid': 1, 'atom': 'C2', 'z': 1},
    },
    'bonds': [{'atoms': ['A', 'B'], 'order': 1}],
}


# ---------------------------------------------------------------------------
# Session-scoped fixtures: run the expensive generation steps once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def sty_workdir(tmp_path_factory):
    """Create a temp working directory and initialise pfs in it.

    pfs_setup with mock=False creates the project sub-directory structure and
    sets projPath (required by grab_files / proj_abspath).  We use projdir='proj'
    so the layout is deterministic; pfs.setup changes cwd to tmpdir/proj/.
    """
    tmpdir = tmp_path_factory.mktemp('sty_parameterize')
    orig = os.getcwd()
    pfs.pfs_setup(root=str(tmpdir), projdir='proj', mock=False)
    # cwd is now tmpdir/proj/  (set by pfs_setup)
    workdir = tmpdir / 'proj'
    yield workdir
    os.chdir(orig)


@pytest.fixture(scope='module')
def sty_molecule(sty_workdir):
    """Generate (parameterise + minimise) the STY monomer.

    STY.mol2 is fetched from the htpolynet system library via pfs.checkout.
    """
    os.chdir(sty_workdir)
    M = Molecule.New('STY', None)
    # Runtime always calls set_sequence_from_moldict before generate(); it sets
    # self.sequence = ['STY'] for a monomer, which generate() later asserts.
    M.set_sequence_from_moldict({'STY': M})
    M.generate(ambertools={'charge_method': 'gas'}, gaff={'minimize_molecules': True})
    return M


@pytest.fixture(scope='module')
def sty1_1_molecule(sty_workdir, sty_molecule):
    """React two STY monomers to form STY1_1.

    Depends on sty_molecule so that STY.top / STY.gro are already present.
    """
    os.chdir(sty_workdir)
    R   = Reaction(_STY1_1_SPEC)
    mol_dict = {'STY': sty_molecule}
    P   = Molecule.New('STY1_1', R)
    P.set_sequence_from_moldict(mol_dict)
    P.generate(
        available_molecules=mol_dict,
        ambertools={'charge_method': 'gas'},
        gaff={'minimize_molecules': True},
    )
    return P


# ---------------------------------------------------------------------------
# Tests — STY parameterisation
# ---------------------------------------------------------------------------

class TestSTYParameterization:

    def test_gro_exists(self, sty_workdir, sty_molecule):
        assert (sty_workdir / 'STY.gro').exists()

    def test_top_exists(self, sty_workdir, sty_molecule):
        assert (sty_workdir / 'STY.top').exists()

    def test_mol2_exists(self, sty_workdir, sty_molecule):
        assert (sty_workdir / 'STY.mol2').exists()

    def test_atom_count(self, sty_molecule):
        """STY.mol2 declares 18 atoms; parameterisation must preserve that."""
        assert sty_molecule.TopoCoord.Coordinates.N == 18

    def test_molecular_weight(self, sty_molecule):
        """Styrene MW is 104.15 g/mol; allow ±2 for rounding."""
        mw = sty_molecule.get_molecular_weight()
        assert 102.0 < mw < 108.0, f'Unexpected MW: {mw:.3f}'

    def test_c1_atom_present(self, sty_molecule):
        names = set(sty_molecule.TopoCoord.Coordinates.A['atomName'])
        assert 'C1' in names

    def test_c2_atom_present(self, sty_molecule):
        names = set(sty_molecule.TopoCoord.Coordinates.A['atomName'])
        assert 'C2' in names


# ---------------------------------------------------------------------------
# Tests — STY1_1 reaction product
# ---------------------------------------------------------------------------

class TestSTY1_1Reaction:

    def test_gro_exists(self, sty_workdir, sty1_1_molecule):
        assert (sty_workdir / 'STY1_1.gro').exists()

    def test_top_exists(self, sty_workdir, sty1_1_molecule):
        assert (sty_workdir / 'STY1_1.top').exists()

    def test_atom_count(self, sty1_1_molecule):
        """Two STY monomers (18 atoms each) bonded with 2 sacrificial H removed: 18+18-2=34."""
        assert sty1_1_molecule.TopoCoord.Coordinates.N == 34

    def test_two_residues(self, sty1_1_molecule):
        A = sty1_1_molecule.TopoCoord.Coordinates.A
        assert A['resNum'].nunique() == 2

    def test_molecular_weight(self, sty1_1_molecule):
        """2 * 104.15 − 2 * 1.008 ≈ 206.28 g/mol; allow ±3."""
        mw = sty1_1_molecule.get_molecular_weight()
        assert 203.0 < mw < 213.0, f'Unexpected MW: {mw:.3f}'

    def test_inter_residue_c1_c2_bond(self, sty1_1_molecule):
        """The new C1(res1)–C2(res2) bond must appear in the GROMACS topology bondlist.

        After GAFF re-parameterisation antechamber may renumber atoms relative to
        the GRO file, so mol2_bondlist indices are unreliable.  The GROMACS
        Topology.bondlist is built from the bonds section of the .top file, whose
        'nr' column is always consistent with the GRO globalIdx.
        """
        A   = sty1_1_molecule.TopoCoord.Coordinates.A
        c1  = int(A.loc[(A['resNum'] == 1) & (A['atomName'] == 'C1'), 'globalIdx'].values[0])
        c2  = int(A.loc[(A['resNum'] == 2) & (A['atomName'] == 'C2'), 'globalIdx'].values[0])
        bl  = sty1_1_molecule.TopoCoord.Topology.bondlist
        assert bl.are_bonded(c1, c2), (
            f'Expected bond between C1(idx={c1}) and C2(idx={c2}) but topology bondlist says otherwise'
        )

    def test_sequence_is_two_sty(self, sty1_1_molecule):
        assert sty1_1_molecule.sequence == ['STY', 'STY']
