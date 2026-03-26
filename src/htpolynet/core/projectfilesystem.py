"""Handles project filesystems.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import shutil
import logging
import os
import glob
import importlib.resources
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_ENV_VAR = 'HTPOLYNET_CACHE'
_CACHE_DEFAULT = Path.home() / '.htpolynet'


class SystemLibrary:
    """Read-only access to bundled package resources via importlib.resources."""

    def __init__(self):
        self._root = importlib.resources.files('htpolynet.resources')

    @property
    def root(self):
        """Filesystem path of the resource root, for display."""
        return str(self._root)

    def exists(self, filename):
        """Checks if filename exists in the system library.

        Args:
            filename (str): path relative to resource root

        Returns:
            bool: True if found
        """
        try:
            return self._root.joinpath(filename).is_file()
        except Exception:
            return False

    def checkout(self, filename):
        """Copies filename from the system library to the current working directory.

        Args:
            filename (str): path relative to resource root

        Returns:
            bool: True if successful
        """
        try:
            src = self._root.joinpath(filename)
            dest = Path(os.getcwd()) / os.path.basename(filename)
            dest.write_bytes(src.read_bytes())
            return True
        except Exception:
            return False

    def get_example_names(self):
        """Returns sorted list of example names available in the depot.

        Returns:
            list: example names without the .tgz extension
        """
        depot = self._root.joinpath('example_depot')
        return sorted(f.name[:-4] for f in depot.iterdir() if f.name.endswith('.tgz'))

    def get_molecule_names(self):
        """Returns sorted list of molecule names available as inputs in the system library.

        Returns:
            list: molecule names (stems of files in molecules/inputs/)
        """
        mol_dir = self._root.joinpath('molecules/inputs')
        return sorted(set(f.name.rsplit('.', 1)[0] for f in mol_dir.iterdir() if not f.name.startswith('.')))

    def get_example_depot_location(self):
        """Returns the filesystem path of the example depot directory.

        Returns:
            str: path to example_depot
        """
        return str(self._root.joinpath('example_depot'))

    def info(self):
        """Returns a description string.

        Returns:
            str: description
        """
        return f'System library is {self.root}'


class UserCache:
    """Writable cache for user-generated files such as parameterized molecules.

    The cache location defaults to ~/.htpolynet but can be overridden by
    setting the HTPOLYNET_CACHE environment variable.
    """

    def __init__(self, path=None):
        """Initializes the user cache, creating the directory if needed.

        Args:
            path (str or Path): cache root; if None uses HTPOLYNET_CACHE env var or ~/.htpolynet
        """
        if path is None:
            path = os.environ.get(_CACHE_ENV_VAR, _CACHE_DEFAULT)
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)

    def exists(self, filename):
        """Checks if filename exists in the cache.

        Args:
            filename (str): path relative to cache root

        Returns:
            bool: True if found
        """
        return (self.root / filename).exists()

    def checkout(self, filename):
        """Copies filename from the cache to the current working directory.

        Args:
            filename (str): path relative to cache root

        Returns:
            bool: True if successful
        """
        src = self.root / filename
        if src.exists():
            shutil.copyfile(src, os.path.basename(filename))
            return True
        return False

    def checkin(self, filename, overwrite=False):
        """Copies a file from the current working directory into the cache.

        Args:
            filename (str): destination path relative to cache root
            overwrite (bool): overwrite if already cached, defaults to False

        Returns:
            bool: False if source not found in cwd; True otherwise
        """
        src = Path(os.path.basename(filename))
        if not src.exists():
            logger.debug(f'{src} not found in {os.getcwd()}. No check-in performed.')
            return False
        dest = self.root / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or overwrite:
            shutil.copyfile(src, dest)
        return True

    def get_molecule_names(self):
        """Returns sorted list of parameterized molecule names in the cache.

        Returns:
            list: molecule names (stems of files in molecules/parameterized/)
        """
        mol_dir = self.root / 'molecules' / 'parameterized'
        if not mol_dir.exists():
            return []
        return sorted(set(f.stem for f in mol_dir.iterdir() if f.is_file()))

    def info(self):
        """Returns a description string.

        Returns:
            str: description
        """
        return f'User cache is {self.root}'


class UserLibrary:
    """User-specified directory of molecule input files."""

    def __init__(self, pathname='.'):
        """Initializes a user library from an existing directory.

        Args:
            pathname (str): path to the user library directory, defaults to '.'
        """
        self.root = Path(os.path.abspath(pathname))
        assert self.root.exists() and self.root.is_dir(), f'{pathname} is not a directory'

    def exists(self, filename):
        """Checks if filename exists in the user library.

        Args:
            filename (str): path relative to library root

        Returns:
            bool: True if found
        """
        return (self.root / filename).exists()

    def checkout(self, filename, searchpath=[], altpath=[]):
        """Copies filename from the user library to the current working directory.

        Args:
            filename (str): path relative to library root
            searchpath (list): additional directories to search, defaults to []
            altpath (list): extra directories to append to the search path, defaults to []

        Returns:
            bool: True if successful
        """
        src = self.root / filename
        if src.exists():
            shutil.copyfile(src, os.path.basename(filename))
            return True
        all_paths = list(searchpath)
        if altpath:
            all_paths.extend(altpath if isinstance(altpath, list) else [altpath])
        for p in all_paths:
            candidate = Path(p) / filename
            if candidate.exists():
                shutil.copyfile(candidate, os.path.basename(filename))
                return True
        return False

    def info(self):
        """Returns a description string.

        Returns:
            str: description
        """
        return f'User library is {self.root}'


_SYSTEM_LIBRARY_: SystemLibrary = None
_USER_CACHE_: UserCache = None


def lib_setup():
    """Sets up the system library and user cache.

    Returns:
        SystemLibrary: the system library object
    """
    global _SYSTEM_LIBRARY_, _USER_CACHE_
    if _SYSTEM_LIBRARY_ is None:
        _SYSTEM_LIBRARY_ = SystemLibrary()
    if _USER_CACHE_ is None:
        _USER_CACHE_ = UserCache()
    return _SYSTEM_LIBRARY_


def system():
    """Returns the system library object.

    Returns:
        SystemLibrary: the system library
    """
    return _SYSTEM_LIBRARY_


class ProjectFileSystem:
    """Handles all aspects of the creation and organization of a project filesystem."""

    def __init__(self, root='.', topdirs=['molecules', 'systems', 'plots'], projdir='next',
                 verbose=False, reProject=False, userlibrary=None, mock=False):
        """Generates a new ProjectFileSystem object.

        Args:
            root (str): path of root directory, defaults to '.'
            topdirs (list): toplevel subdirectory names, defaults to ['molecules','systems','plots']
            projdir (str): project directory name or 'next', defaults to 'next'
            verbose (bool): verbose output flag, defaults to False
            reProject (bool): restart flag, defaults to False
            userlibrary (str): path to user library directory, defaults to None
            mock (bool): mock call flag, defaults to False
        """
        lib_setup()
        self.userlibrary = UserLibrary(userlibrary) if userlibrary else None
        self.rootPath = os.path.abspath(root)
        os.chdir(self.rootPath)
        self.cwd = self.rootPath
        self.verbose = verbose
        if not mock:
            self._next_project_dir(projdir=projdir, reProject=reProject)
            self._setup_project_dir(topdirs=topdirs)

    def cdroot(self):
        """Changes the cwd to the root directory."""
        os.chdir(self.rootPath)
        self.cwd = self.rootPath

    def cdproj(self):
        """Changes the cwd to the toplevel project directory."""
        os.chdir(self.projPath)
        self.cwd = self.projPath

    def go_to(self, subPath, make=False):
        """Changes the cwd to the directory named by subPath.

        Args:
            subPath (str): directory relative to project directory
            make (bool): create directory if missing, defaults to False
        """
        self.cdproj()
        if os.path.exists(subPath):
            os.chdir(subPath)
            self.cwd = os.getcwd()
        elif make:
            os.mkdir(subPath)
            os.chdir(subPath)
            self.cwd = os.getcwd()

    def __str__(self):
        return f'root {self.rootPath}: cwd {self.cwd}'

    def _next_project_dir(self, projdir='next', reProject=False, prefix='proj-'):
        """Determines and creates the project directory.

        Args:
            projdir (str): explicit name or 'next', defaults to 'next'
            reProject (bool): restart flag, defaults to False
            prefix (str): prefix for auto-named directories, defaults to 'proj-'
        """
        if projdir != 'next':
            self.projPath = os.path.join(self.rootPath, projdir)
            if os.path.exists(projdir):
                logger.info(f'Working in existing project {self.projPath}')
            else:
                os.mkdir(projdir)
                logger.info(f'Working in new project {self.projPath}')
        else:
            i = 0
            lastprojdir = ''
            while os.path.isdir(os.path.join(self.rootPath, f'{prefix}{i}')):
                lastprojdir = f'{prefix}{i}'
                logger.debug(f'{lastprojdir} exists')
                i += 1
            assert not os.path.exists(f'{prefix}{i}')
            if not reProject or lastprojdir == '':
                currentprojdir = f'{prefix}0' if lastprojdir == '' else f'{prefix}{i}'
                self.projPath = os.path.join(self.rootPath, currentprojdir)
                logger.info(f'New project in {self.projPath}')
                os.mkdir(currentprojdir)
            else:
                self.projPath = os.path.join(self.rootPath, lastprojdir)
                logger.info(f'Restarting project in {self.projPath} (latest project)')

    def _setup_project_dir(self, topdirs=['molecules', 'systems', 'plots']):
        """Creates toplevel subdirectories within the project directory.

        Args:
            topdirs (list): subdirectory names to create, defaults to ['molecules','systems','plots']
        """
        os.chdir(self.projPath)
        self.projSubPaths = {}
        for tops in topdirs:
            self.projSubPaths[tops] = os.path.join(self.projPath, tops)
            if not os.path.isdir(self.projSubPaths[tops]):
                os.mkdir(tops)


_PFS_: ProjectFileSystem = None


def pfs_setup(root='.', topdirs=['molecules', 'systems', 'plots'], projdir='next',
              verbose=False, reProject=False, userlibrary=None, mock=False):
    """Sets up the global ProjectFileSystem.

    Args:
        root (str): parent directory, defaults to '.'
        topdirs (list): toplevel subdirectories, defaults to ['molecules','systems','plots']
        projdir (str): project directory name, defaults to 'next'
        verbose (bool): verbose flag, defaults to False
        reProject (bool): restart flag, defaults to False
        userlibrary (str): user library path, defaults to None
        mock (bool): mock flag, defaults to False
    """
    global _PFS_
    _PFS_ = ProjectFileSystem(root=root, topdirs=topdirs, projdir=projdir,
                               verbose=verbose, reProject=reProject,
                               userlibrary=userlibrary, mock=mock)


def checkout(filename, altpath=[]):
    """Copies a file to cwd; searches user library, then user cache, then system library.

    Args:
        filename (str): path relative to library root
        altpath (list): extra search directories, defaults to []

    Returns:
        bool: True if checkout was successful
    """
    if _PFS_ and _PFS_.userlibrary and _PFS_.userlibrary.checkout(
            filename, searchpath=[_PFS_.rootPath, _PFS_.projPath], altpath=altpath):
        return True
    if _USER_CACHE_.checkout(filename):
        return True
    return _SYSTEM_LIBRARY_.checkout(filename)


def checkin(filename, overwrite=False):
    """Checks a file from cwd into the user cache.

    Args:
        filename (str): destination path relative to cache root
        overwrite (bool): overwrite if already cached, defaults to False
    """
    _USER_CACHE_.checkin(filename, overwrite=overwrite)


def fetch_molecule_files(mname):
    """Fetches all relevant molecule data files for the named molecule.

    Args:
        mname (str): molecule name

    Returns:
        list: file extensions found and fetched
    """
    ret_exts = []
    dirname = 'molecules/parameterized'
    for e in ['mol2', 'pdb', 'gro', 'top', 'tpx', 'itp', 'grx']:
        prob_filename = os.path.join(dirname, f'{mname}.{e}')
        if exists(prob_filename):
            ret_exts.append(e)
            checkout(prob_filename)
    return ret_exts


def exists(filename):
    """Checks for filename in user library, user cache, then system library.

    Args:
        filename (str): path relative to library root

    Returns:
        bool: True if found anywhere
    """
    if _PFS_ and _PFS_.userlibrary and _PFS_.userlibrary.exists(filename):
        return True
    if _USER_CACHE_.exists(filename):
        return True
    return _SYSTEM_LIBRARY_.exists(filename)


def subpath(name):
    """Returns the path of the named project subdirectory.

    Args:
        name (str): subdirectory name

    Returns:
        str: path of subdirectory
    """
    return _PFS_.projSubPaths[name]


def go_proj():
    """Changes the current working directory to the project directory."""
    _PFS_.cdproj()


def go_root():
    """Changes the current working directory to the root directory."""
    _PFS_.cdroot()


def go_to(pathstr):
    """Changes the current working directory to pathstr relative to the project root.

    Args:
        pathstr (str): directory path relative to project root

    Returns:
        bool: True if the directory already existed, False if newly created
    """
    _PFS_.cdproj()
    dirname = os.path.dirname(pathstr)
    if dirname == '':
        dirname = pathstr
    assert dirname in _PFS_.projSubPaths, f'Error: cannot navigate using pathstring {pathstr}'
    reentry = True
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        reentry = False
    os.chdir(dirname)
    basename = os.path.basename(pathstr)
    if basename != pathstr:
        if not os.path.exists(basename):
            logger.debug(f'PFS: making {basename}')
            os.mkdir(basename)
        os.chdir(basename)
    _PFS_.cwd = os.getcwd()
    return reentry


def root():
    """Returns the root directory path.

    Returns:
        str: root directory path
    """
    return _PFS_.rootPath


def cwd():
    """Returns the current working directory relative to the root path.

    Returns:
        str: relative path of current working directory
    """
    return os.path.relpath(os.getcwd(), start=_PFS_.rootPath)


def proj():
    """Returns the project directory path.

    Returns:
        str: project directory path
    """
    return _PFS_.projPath


def local_data_searchpath():
    """Returns root and project paths for local data searches.

    Returns:
        list: [rootPath, projPath]
    """
    return [_PFS_.rootPath, _PFS_.projPath]


def get_molecule_info():
    """Returns molecule names available in the system library and user cache.

    Returns:
        tuple: (system_molecules, cached_molecules) each a sorted list of names
    """
    system_mols = _SYSTEM_LIBRARY_.get_molecule_names() if _SYSTEM_LIBRARY_ else []
    cached_mols = _USER_CACHE_.get_molecule_names() if _USER_CACHE_ else []
    return system_mols, cached_mols


def info():
    """Prints summary information about active libraries to the console."""
    if _PFS_ and _PFS_.userlibrary:
        print(_PFS_.userlibrary.info())
    if _USER_CACHE_:
        print(_USER_CACHE_.info())
    if _SYSTEM_LIBRARY_:
        print(_SYSTEM_LIBRARY_.info())


def proj_abspath(filename):
    """Returns the path of filename relative to the project directory.

    Args:
        filename (str): filename to resolve

    Returns:
        str: path relative to the project directory
    """
    abf = os.path.abspath(filename)
    return os.path.relpath(abf, _PFS_.projPath)
