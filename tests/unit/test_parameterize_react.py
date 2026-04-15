"""Integration tests for molecule parameterisation and inter-monomer reaction.

These tests exercise the full AmberTools + GROMACS pipeline:
  1. Parameterise the STY monomer (antechamber → tleap → parmed → gmx).
  2. React two STY monomers to form the STY1_1 dimer (merge, bond, re-parameterise).
  3. Expand chain reactions to generate the STY1_1~C1=C2~STY trimer template
     and verify that its BondTemplate matches the pattern needed for CURE iteration 2.

All external tools (antechamber, tleap, parmchk2, gmx) are expected to be on
PATH — conftest.py prepends the active conda environment's bin dir automatically.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import os
import pytest

from htpolynet.core.molecule import Molecule
from htpolynet.core.bondtemplate import BondTemplate
from htpolynet.core.topocoord import find_template
from htpolynet.cure.expandreactions import bondchain_expand_reactions
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

    zrecs must be set before generate() so that initialize_monomer_grx_attributes
    can locate the two reactive carbons (C1 and C2) and build the ChainManager.
    In the real runtime these are populated via Molecule.update_zrecs() from the
    reaction spec; here we supply them directly.
    """
    os.chdir(sty_workdir)
    M = Molecule.New('STY', None)
    # Runtime always calls set_sequence_from_moldict before generate(); it sets
    # self.sequence = ['STY'] for a monomer, which generate() later asserts.
    M.set_sequence_from_moldict({'STY': M})
    # Reactive atoms: C1 (head) and C2 (tail) with z=1, derived from _STY1_1_SPEC
    M.zrecs = [
        {'resid': 1, 'atom': 'C1', 'z': 1},
        {'resid': 1, 'atom': 'C2', 'z': 1},
    ]
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

    def test_chain_manager_has_four_atom_chain(self, sty1_1_molecule):
        """STY1_1's ChainManager must carry exactly one 4-atom bondchain.

        After bonding C1(STY-1) to C2(STY-2) the two 2-element monomer chains
        merge into [C1_r2, C2_r2, C1_r1, C2_r1] — a 4-atom chain.  This chain
        is required by bondchain_expand_reactions to classify STY1_1 as a
        'dimer' and generate the trimer template reaction.
        """
        cm = sty1_1_molecule.chain_manager
        assert len(cm.chains) == 1, (
            f'Expected 1 chain in STY1_1 chain_manager, found {len(cm.chains)}'
        )
        chain_len = len(cm.chains[0].idx_list)
        assert chain_len == 4, (
            f'Expected chain length 4 in STY1_1, found {chain_len}'
        )


# ---------------------------------------------------------------------------
# Fixture — trimer 'STY1_1~C1=C2~STY' generated via bondchain_expand_reactions
# ---------------------------------------------------------------------------

# All three chain-expansion products expected for the STY+STY1_1 system
_EXPECTED_CHAIN_PRODUCTS = {
    'STY~C1=C2~STY1_1',      # monomer head attacks dimer tail
    'STY1_1~C1=C2~STY',      # dimer head attacks monomer tail
    'STY1_1~C1=C2~STY1_1',   # dimer head attacks dimer tail
}
_TRIMER_NAME = 'STY1_1~C1=C2~STY'

@pytest.fixture(scope='module')
def chain_expanded_molecules(sty_workdir, sty_molecule, sty1_1_molecule):
    """Run bondchain_expand_reactions and return the full extra_molecules dict.

    All three chain-expansion products must be present before any generation
    so that tests can assert on the full set independently of generation order.
    """
    os.chdir(sty_workdir)
    mol_dict = {'STY': sty_molecule, 'STY1_1': sty1_1_molecule}
    _extra_reactions, extra_molecules = bondchain_expand_reactions(mol_dict)
    return extra_molecules


@pytest.fixture(scope='module')
def trimer_molecule(sty_workdir, sty_molecule, sty1_1_molecule, chain_expanded_molecules):
    """Generate the 'STY1_1~C1=C2~STY' trimer template.

    bondchain_expand_reactions inspects monomer and dimer chain_managers and
    creates the reactions (and shell Molecule objects) needed for trimers.
    Here we take the 'STY1_1~C1=C2~STY' shell and drive it through generate()
    so that its bond_templates are populated for matching tests.
    """
    os.chdir(sty_workdir)
    mol_dict = {'STY': sty_molecule, 'STY1_1': sty1_1_molecule}
    trimer = chain_expanded_molecules[_TRIMER_NAME]
    trimer.generate(
        available_molecules=mol_dict,
        ambertools={'charge_method': 'gas'},
        gaff={'minimize_molecules': True},
    )
    return trimer


# ---------------------------------------------------------------------------
# Tests — chain expansion completeness
# ---------------------------------------------------------------------------

class TestChainExpansion:
    """Verify that bondchain_expand_reactions produces all three required oligomers."""

    def test_all_three_products_generated(self, chain_expanded_molecules):
        missing = _EXPECTED_CHAIN_PRODUCTS - set(chain_expanded_molecules.keys())
        assert not missing, (
            f'bondchain_expand_reactions is missing products: {missing}\n'
            f'produced: {list(chain_expanded_molecules.keys())}'
        )

    def test_monomer_head_attacks_dimer_tail(self, chain_expanded_molecules):
        """STY(C1) attacks C2 of the first STY in STY1_1."""
        assert 'STY~C1=C2~STY1_1' in chain_expanded_molecules

    def test_dimer_head_attacks_monomer_tail(self, chain_expanded_molecules):
        """C1 of the second STY in STY1_1 attacks STY(C2)."""
        assert 'STY1_1~C1=C2~STY' in chain_expanded_molecules

    def test_dimer_head_attacks_dimer_tail(self, chain_expanded_molecules):
        """C1 of second STY in one STY1_1 attacks C2 of first STY in another STY1_1."""
        assert 'STY1_1~C1=C2~STY1_1' in chain_expanded_molecules

    def test_exactly_three_products(self, chain_expanded_molecules):
        assert len(chain_expanded_molecules) == 3, (
            f'Expected exactly 3 chain-expanded products, got {len(chain_expanded_molecules)}: '
            f'{list(chain_expanded_molecules.keys())}'
        )


# ---------------------------------------------------------------------------
# Tests — trimer generation and BondTemplate matching
# ---------------------------------------------------------------------------

class TestTrimerBondTemplate:

    def test_trimer_generated(self, trimer_molecule):
        assert trimer_molecule.name == _TRIMER_NAME

    def test_trimer_atom_count(self, trimer_molecule):
        """STY1_1 (34 atoms) + STY (18 atoms) − 2 sacrificial H = 50 atoms."""
        assert trimer_molecule.TopoCoord.Coordinates.N == 50

    def test_trimer_three_residues(self, trimer_molecule):
        A = trimer_molecule.TopoCoord.Coordinates.A
        assert A['resNum'].nunique() == 3

    def test_trimer_has_bond_template(self, trimer_molecule):
        """generate() must call prepare_new_bonds, leaving at least one BondTemplate."""
        assert len(trimer_molecule.bond_templates) >= 1, (
            'Trimer has no bond_templates — prepare_new_bonds may not have run'
        )

    def test_trimer_bond_template_oneaway_resnames(self, trimer_molecule):
        """The trimer's BondTemplate must carry oneaway_resnames=['STY', None].

        This is the pattern that arises in CURE iteration 2 when the head of a
        growing 4-atom chain bonds to a new monomer tail.  The trimer is the
        smallest molecule that expresses this context and must supply the
        matching template so that map_from_templates can assign GAFF parameters.
        """
        bt = trimer_molecule.bond_templates[0]
        assert bt.oneaway_resnames == ['STY', None], (
            f'Expected oneaway_resnames [STY, None], got {bt.oneaway_resnames}'
        )

    def test_trimer_bond_template_oneaway_atomnames(self, trimer_molecule):
        bt = trimer_molecule.bond_templates[0]
        assert bt.oneaway_atomnames == ['C1', None], (
            f'Expected oneaway_atomnames [C1, None], got {bt.oneaway_atomnames}'
        )

    def test_find_template_for_cure2_bond(self, trimer_molecule):
        """find_template must succeed for the BondTemplate that fails in CURE iteration 2.

        The failing pattern (from the ERROR log) is:
            BondTemplate ['C1', 'C2'] resnames ['STY', 'STY'] intraresidue? False
            order 1 bystander-resnames [[], []] bystander-atomnames [[], []]
            oneaway-resnames ['STY', None] oneaway-atomnames ['C1', None]

        With the trimer in the moldict, find_template must return without raising.
        """
        cure2_bt = BondTemplate(
            names=['C1', 'C2'],
            resnames=['STY', 'STY'],
            intraresidue=False,
            order=1,
            bystander_resnames=[[], []],
            bystander_atomnames=[[], []],
            oneaway_resnames=['STY', None],
            oneaway_atomnames=['C1', None],
        )
        moldict = {_TRIMER_NAME: trimer_molecule}
        # find_template raises if no match; success means the template is present
        template_mol, _rb, _rev = find_template(cure2_bt, moldict)
        assert template_mol.name == _TRIMER_NAME
