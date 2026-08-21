"""Tests the bundled system resource library.

Rewritten against ``SystemLibrary``.  The previous version imported
``RuntimeLibrary`` from ``htpolynet.utils.projectfilesystem`` and ``Software``
from ``htpolynet.external.software`` -- both removed in the 2.0 refactor, and
neither module path still exists.  Because the bad imports were at module
scope, this file aborted collection for the entire unit suite.

.. moduleauthor: Cameron F. Abrams, <cfa22@drexel.edu>
"""
import os

import pytest

from htpolynet.core.projectfilesystem import SystemLibrary


@pytest.fixture
def syslib():
    return SystemLibrary()


def test_root_is_a_directory(syslib):
    assert os.path.isdir(syslib.root)


@pytest.mark.parametrize('subdir', ['example_depot', 'mdp', 'molecules', 'tcl'])
def test_bundled_resource_subdirs_present(syslib, subdir):
    assert os.path.isdir(os.path.join(syslib.root, subdir))


def test_example_depot_location_is_a_directory(syslib):
    assert os.path.isdir(syslib.get_example_depot_location())


def test_example_names_cover_the_shipped_depot(syslib):
    names = syslib.get_example_names()
    assert '0-liquid-styrene' in names
    assert '6-cyanate-ester' in names
    assert names == sorted(names), 'depot names should come back in numeric-prefix order'


def test_example_names_carry_no_extension(syslib):
    assert not any('.' in n.rsplit('-', 1)[-1] for n in syslib.get_example_names())


def test_exists_finds_a_bundled_file(syslib):
    assert syslib.exists('mdp/min.mdp')


def test_exists_rejects_a_missing_file(syslib):
    assert not syslib.exists('mdp/no-such-file.mdp')


def test_exists_is_a_file_test_not_a_path_test(syslib):
    """``mdp`` is a real directory, but ``exists`` is documented as a file test."""
    assert not syslib.exists('mdp')


def test_molecule_names_are_nonempty(syslib):
    assert len(syslib.get_molecule_names()) > 0


def test_checkout_copies_into_the_current_directory(syslib, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert syslib.checkout('mdp/min.mdp')
    assert (tmp_path / 'min.mdp').is_file()


def test_checkout_reports_failure_for_a_missing_file(syslib, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not syslib.checkout('mdp/no-such-file.mdp')


def test_info_mentions_the_root(syslib):
    assert syslib.root in syslib.info()
