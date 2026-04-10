"""

.. module:: conftest.py
   :synopsis: pytest configuration
   
.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>

"""
import pytest
import os
from pathlib import Path
# import sys
collect_ignore = ["__defunct__"]

def pytest_configure(config):
    """Ensure the running Python's bin dir is on PATH.

    When invoked as ``/path/to/env/bin/python -m pytest``, the env's bin
    directory is not always inherited by subprocesses.  Prepending it makes
    co-installed tools (e.g. gmx) discoverable without hard-coding any path.
    """
    import sys
    env_bin = os.path.dirname(sys.executable)
    path = os.environ.get('PATH', '')
    if env_bin not in path:
        os.environ['PATH'] = env_bin + os.pathsep + path
# if sys.version_info[0] > 2:
#     collect_ignore.append("pkg/module_py2.py")


def pytest_addoption(parser):
    parser.addoption(
        '--gen-figs', action='store_true', default=False,
        help='Generate documentation figures as test side effects'
    )


@pytest.fixture
def figdir(request):
    """Return a Path to the figure output dir, or None when --gen-figs is absent.

    Tests that produce documentation figures should call
    ``pytest.skip(...)`` when this fixture returns None.
    """
    if not request.config.getoption('--gen-figs'):
        return None
    d = (Path(__file__).parent.parent
         / 'docs' / 'source' / 'user-guide' / 'pics' / 'ring_pierce')
    d.mkdir(parents=True, exist_ok=True)
    return d

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """Causes each test to run in the directory in which the module is found **or** a subdirectory with the same base name as the module

    For example, say you have <packagename>/tests/unit/test_foo.py.  If the directory
    <packagename>/tests/unit/test_foo/ exists, the test will run in that directory; if not, it will run in <packagename>/tests/unit/

    :param request: pytest request
    :type request: pytest request
    :param monkeypatch: pytest monkeypatch
    :type monkeypatch: pytest.Monkeypatch
    """
    module_bn=request.fspath.basename
    module_name,_=os.path.splitext(module_bn)
    # print('module name',module_name)
    test_dir=request.fspath.dirname
    subdir=os.path.join(test_dir,module_name)
    if os.path.isdir(subdir):
        monkeypatch.chdir(subdir)
    else:
        monkeypatch.chdir(test_dir)
