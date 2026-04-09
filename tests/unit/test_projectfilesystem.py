"""

.. module:: test_projectfilesystem
   :synopsis: unit tests for htpolynet.core.projectfilesystem

.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import unittest
import os
import tempfile
import shutil
import logging
logger = logging.getLogger(__name__)

import htpolynet.core.projectfilesystem as pfs
from htpolynet.core.projectfilesystem import (
    SystemLibrary, UserCache, UserLibrary,
    ProjectFileSystem, Dirs,
    lib_setup, system,
    checkout, checkin, exists, fetch_molecule_files,
    go_proj, go_root, go_to, subpath,
    root, proj, cwd, proj_abspath,
    local_data_searchpath, get_molecule_info, info,
    pfs_setup,
)


class TestDirs(unittest.TestCase):
    """Dirs namespace — all path strings should be stable and consistent."""

    def test_molecules_paths(self):
        self.assertEqual(Dirs.molecules, 'molecules')
        self.assertEqual(Dirs.molecules_inputs, 'molecules/inputs')
        self.assertEqual(Dirs.molecules_parameterized, 'molecules/parameterized')
        self.assertTrue(Dirs.molecules_parameterized.startswith(Dirs.molecules + '/'))

    def test_systems_paths(self):
        self.assertEqual(Dirs.systems, 'systems')
        for attr in ('systems_init', 'systems_densification', 'systems_precure',
                     'systems_postcure', 'systems_capping', 'systems_final'):
            val = getattr(Dirs, attr)
            self.assertTrue(val.startswith('systems/'), f'{attr} should start with systems/')

    def test_systems_iter(self):
        self.assertEqual(Dirs.systems_iter(0), 'systems/iter-0')
        self.assertEqual(Dirs.systems_iter(7), 'systems/iter-7')

    def test_mdp_file(self):
        self.assertEqual(Dirs.mdp_file('npt'), 'mdp/npt.mdp')
        self.assertEqual(Dirs.mdp_file('nvt'), 'mdp/nvt.mdp')
        self.assertTrue(Dirs.mdp_file('min').endswith('.mdp'))

    def test_topdirs_lists(self):
        self.assertIn('molecules', Dirs.run_topdirs)
        self.assertIn('systems', Dirs.run_topdirs)
        self.assertIn('plots', Dirs.run_topdirs)
        self.assertIn('postsim', Dirs.postsim_topdirs)
        self.assertIn('analyze', Dirs.analyze_topdirs)
        # each list is a superset of the previous
        for d in Dirs.run_topdirs:
            self.assertIn(d, Dirs.postsim_topdirs)
        for d in Dirs.postsim_topdirs:
            self.assertIn(d, Dirs.analyze_topdirs)


class TestSystemLibrary(unittest.TestCase):
    """SystemLibrary — read-only access to bundled package resources."""

    def setUp(self):
        self.lib = SystemLibrary()

    def test_root_exists(self):
        self.assertTrue(os.path.isdir(self.lib.root))

    def test_example_depot_not_empty(self):
        names = self.lib.get_example_names()
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)

    def test_example_names_sorted(self):
        names = self.lib.get_example_names()
        self.assertEqual(names, sorted(names))

    def test_example_names_no_extension(self):
        for name in self.lib.get_example_names():
            self.assertFalse(name.endswith('.tgz'))

    def test_molecule_names(self):
        names = self.lib.get_molecule_names()
        self.assertIsInstance(names, list)

    def test_exists_known_mdp(self):
        # every installation must have at least one mdp file
        self.assertTrue(self.lib.exists('mdp/npt.mdp') or
                        self.lib.exists('mdp/nvt.mdp') or
                        self.lib.exists('mdp/min.mdp'))

    def test_exists_missing(self):
        self.assertFalse(self.lib.exists('does/not/exist.xyz'))

    def test_checkout_missing_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = os.getcwd()
            os.chdir(tmp)
            try:
                result = self.lib.checkout('does/not/exist.xyz')
                self.assertFalse(result)
            finally:
                os.chdir(orig)

    def test_get_example_depot_location(self):
        loc = self.lib.get_example_depot_location()
        self.assertIsInstance(loc, str)
        self.assertTrue(os.path.isdir(loc))

    def test_info_string(self):
        s = self.lib.info()
        self.assertIn('System library', s)


class TestUserCache(unittest.TestCase):
    """UserCache — writable cache backed by a temporary directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = UserCache(path=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_root_created(self):
        self.assertTrue(os.path.isdir(self.cache.root))

    def test_exists_missing(self):
        self.assertFalse(self.cache.exists('no/such/file.txt'))

    def test_checkin_and_checkout(self):
        with tempfile.TemporaryDirectory() as workdir:
            # create a source file in workdir
            src = os.path.join(workdir, 'mol.gro')
            with open(src, 'w') as f:
                f.write('test gro content')
            orig = os.getcwd()
            os.chdir(workdir)
            try:
                result = self.cache.checkin('molecules/parameterized/mol.gro')
                self.assertTrue(result)
                self.assertTrue(self.cache.exists('molecules/parameterized/mol.gro'))

                # checkout into a fresh directory
                dest_dir = tempfile.mkdtemp()
                os.chdir(dest_dir)
                try:
                    ok = self.cache.checkout('molecules/parameterized/mol.gro')
                    self.assertTrue(ok)
                    self.assertTrue(os.path.exists('mol.gro'))
                    with open('mol.gro') as f:
                        self.assertEqual(f.read(), 'test gro content')
                finally:
                    os.chdir(orig)
                    shutil.rmtree(dest_dir, ignore_errors=True)
            finally:
                os.chdir(orig)

    def test_checkin_missing_source_returns_false(self):
        with tempfile.TemporaryDirectory() as workdir:
            orig = os.getcwd()
            os.chdir(workdir)
            try:
                result = self.cache.checkin('molecules/parameterized/ghost.gro')
                self.assertFalse(result)
            finally:
                os.chdir(orig)

    def test_checkin_no_overwrite(self):
        with tempfile.TemporaryDirectory() as workdir:
            orig = os.getcwd()
            os.chdir(workdir)
            try:
                fname = 'mol.top'
                with open(fname, 'w') as f:
                    f.write('version 1')
                self.cache.checkin(f'molecules/parameterized/{fname}')
                with open(fname, 'w') as f:
                    f.write('version 2')
                self.cache.checkin(f'molecules/parameterized/{fname}', overwrite=False)
                cached = self.cache.root / 'molecules' / 'parameterized' / fname
                with open(cached) as f:
                    self.assertEqual(f.read(), 'version 1')
            finally:
                os.chdir(orig)

    def test_checkin_with_overwrite(self):
        with tempfile.TemporaryDirectory() as workdir:
            orig = os.getcwd()
            os.chdir(workdir)
            try:
                fname = 'mol.top'
                with open(fname, 'w') as f:
                    f.write('version 1')
                self.cache.checkin(f'molecules/parameterized/{fname}')
                with open(fname, 'w') as f:
                    f.write('version 2')
                self.cache.checkin(f'molecules/parameterized/{fname}', overwrite=True)
                cached = self.cache.root / 'molecules' / 'parameterized' / fname
                with open(cached) as f:
                    self.assertEqual(f.read(), 'version 2')
            finally:
                os.chdir(orig)

    def test_checkout_missing_returns_false(self):
        with tempfile.TemporaryDirectory() as workdir:
            orig = os.getcwd()
            os.chdir(workdir)
            try:
                result = self.cache.checkout('molecules/parameterized/ghost.gro')
                self.assertFalse(result)
            finally:
                os.chdir(orig)

    def test_get_molecule_names_empty(self):
        names = self.cache.get_molecule_names()
        self.assertEqual(names, [])

    def test_get_molecule_names_after_checkin(self):
        with tempfile.TemporaryDirectory() as workdir:
            orig = os.getcwd()
            os.chdir(workdir)
            try:
                for mol in ('MOL', 'SOL'):
                    with open(f'{mol}.gro', 'w') as f:
                        f.write('x')
                    self.cache.checkin(f'molecules/parameterized/{mol}.gro')
                names = self.cache.get_molecule_names()
                self.assertIn('MOL', names)
                self.assertIn('SOL', names)
            finally:
                os.chdir(orig)

    def test_info_string(self):
        s = self.cache.info()
        self.assertIn('User cache', s)


class TestUserLibrary(unittest.TestCase):
    """UserLibrary — user-managed directory of input files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lib = UserLibrary(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_root_set(self):
        self.assertEqual(str(self.lib.root), os.path.abspath(self.tmpdir))

    def test_bad_path_raises(self):
        with self.assertRaises(AssertionError):
            UserLibrary('/this/path/does/not/exist')

    def test_exists_missing(self):
        self.assertFalse(self.lib.exists('no/such/file.mol2'))

    def test_checkout_present(self):
        src = os.path.join(self.tmpdir, 'molecules', 'inputs')
        os.makedirs(src)
        with open(os.path.join(src, 'ETH.mol2'), 'w') as f:
            f.write('mol2 data')

        with tempfile.TemporaryDirectory() as workdir:
            orig = os.getcwd()
            os.chdir(workdir)
            try:
                ok = self.lib.checkout('molecules/inputs/ETH.mol2')
                self.assertTrue(ok)
                self.assertTrue(os.path.exists('ETH.mol2'))
            finally:
                os.chdir(orig)

    def test_checkout_missing_returns_false(self):
        with tempfile.TemporaryDirectory() as workdir:
            orig = os.getcwd()
            os.chdir(workdir)
            try:
                ok = self.lib.checkout('molecules/inputs/GHOST.mol2')
                self.assertFalse(ok)
            finally:
                os.chdir(orig)

    def test_info_string(self):
        s = self.lib.info()
        self.assertIn('User library', s)


class TestLibSetup(unittest.TestCase):
    """lib_setup() and system() — global singleton initialisation."""

    def test_lib_setup_returns_system_library(self):
        sl = lib_setup()
        self.assertIsInstance(sl, SystemLibrary)

    def test_system_returns_same_instance(self):
        lib_setup()
        sl = system()
        self.assertIsInstance(sl, SystemLibrary)

    def test_lib_setup_idempotent(self):
        first = lib_setup()
        second = lib_setup()
        self.assertIs(first, second)


class TestProjectFileSystem(unittest.TestCase):
    """ProjectFileSystem and module-level helpers — navigation and file ops."""

    def setUp(self):
        self.base = tempfile.mkdtemp()
        # lib_setup must be called before pfs_setup
        lib_setup()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _setup(self, projdir='next', topdirs=None):
        topdirs = topdirs or Dirs.run_topdirs
        pfs_setup(root=self.base, topdirs=topdirs, projdir=projdir,
                  verbose=False, reProject=False, userlibrary=None)

    def test_first_project_created(self):
        self._setup()
        self.assertTrue(os.path.isdir(os.path.join(self.base, 'proj-0')))

    def test_second_project_increments(self):
        self._setup()          # proj-0
        self._setup()          # proj-1
        self.assertTrue(os.path.isdir(os.path.join(self.base, 'proj-1')))

    def test_named_project(self):
        self._setup(projdir='my-run')
        self.assertTrue(os.path.isdir(os.path.join(self.base, 'my-run')))

    def test_topdirs_created(self):
        self._setup()
        proj_path = os.path.join(self.base, 'proj-0')
        for d in Dirs.run_topdirs:
            self.assertTrue(os.path.isdir(os.path.join(proj_path, d)))

    def test_root_returns_base(self):
        self._setup()
        self.assertEqual(root(), self.base)

    def test_proj_returns_proj_path(self):
        self._setup()
        self.assertEqual(proj(), os.path.join(self.base, 'proj-0'))

    def test_go_proj_changes_cwd(self):
        self._setup()
        go_root()
        go_proj()
        self.assertEqual(os.getcwd(), proj())

    def test_go_root_changes_cwd(self):
        self._setup()
        go_proj()
        go_root()
        self.assertEqual(os.getcwd(), root())

    def test_go_to_creates_subdir(self):
        self._setup()
        go_to('systems/init')
        self.assertEqual(os.getcwd(), os.path.join(proj(), 'systems', 'init'))
        self.assertTrue(os.path.isdir(os.getcwd()))

    def test_go_to_returns_reentry_false_first_time(self):
        self._setup()
        reentry = go_to('systems/densification')
        self.assertFalse(reentry)

    def test_go_to_returns_reentry_true_second_time(self):
        self._setup()
        go_to('systems/precure')
        go_root()
        reentry = go_to('systems/precure')
        self.assertTrue(reentry)

    def test_subpath_returns_existing_dir(self):
        self._setup()
        p = subpath('molecules')
        self.assertTrue(os.path.isdir(p))

    def test_cwd_is_relative(self):
        self._setup()
        go_proj()
        rel = cwd()
        self.assertFalse(os.path.isabs(rel))

    def test_local_data_searchpath(self):
        self._setup()
        paths = local_data_searchpath()
        self.assertIn(root(), paths)
        self.assertIn(proj(), paths)

    def test_proj_abspath(self):
        self._setup()
        go_to('systems/init')
        # write a dummy file here
        with open('dummy.gro', 'w') as f:
            f.write('x')
        rel = proj_abspath('dummy.gro')
        # should be relative to proj, so starts with systems/
        self.assertTrue(rel.startswith('systems'))

    def test_checkout_from_system_library(self):
        """checkout() should retrieve a known mdp from the system library."""
        self._setup()
        go_to('systems/init')
        # npt.mdp is always present in the system library
        result = checkout('mdp/npt.mdp')
        self.assertTrue(result)
        self.assertTrue(os.path.exists('npt.mdp'))

    def test_exists_from_system_library(self):
        self._setup()
        self.assertTrue(exists('mdp/npt.mdp'))
        self.assertFalse(exists('mdp/ghost_that_does_not_exist.mdp'))

    def test_checkin_roundtrip(self):
        self._setup()
        go_to(Dirs.molecules_parameterized)
        fname = 'TESTMOL.gro'
        with open(fname, 'w') as f:
            f.write('test gro')
        checkin(f'{Dirs.molecules_parameterized}/{fname}')
        # now remove local copy and check it can be fetched back
        os.remove(fname)
        result = checkout(f'{Dirs.molecules_parameterized}/{fname}')
        self.assertTrue(result)
        self.assertTrue(os.path.exists(fname))

    def test_get_molecule_info(self):
        self._setup()
        sys_mols, cached_mols = get_molecule_info()
        self.assertIsInstance(sys_mols, list)
        self.assertIsInstance(cached_mols, list)

    def test_info_does_not_raise(self):
        self._setup()
        try:
            info()
        except Exception as e:
            self.fail(f'info() raised {e}')


class TestDirsIntegration(unittest.TestCase):
    """Verify Dirs constants work correctly as pfs_setup topdirs."""

    def setUp(self):
        self.base = tempfile.mkdtemp()
        lib_setup()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_postsim_topdirs_all_created(self):
        pfs_setup(root=self.base, topdirs=Dirs.postsim_topdirs,
                  projdir='p', verbose=False)
        proj_path = os.path.join(self.base, 'p')
        for d in Dirs.postsim_topdirs:
            self.assertTrue(os.path.isdir(os.path.join(proj_path, d)))

    def test_analyze_topdirs_all_created(self):
        pfs_setup(root=self.base, topdirs=Dirs.analyze_topdirs,
                  projdir='p', verbose=False)
        proj_path = os.path.join(self.base, 'p')
        for d in Dirs.analyze_topdirs:
            self.assertTrue(os.path.isdir(os.path.join(proj_path, d)))
