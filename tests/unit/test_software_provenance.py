"""Tests what an htpolynet build can say about the code that produced it.

`htpolynet info` reports a git commit, which works from a source checkout and
does not exist for an installed copy -- pip, conda, or the published
container, none of which carry a `.git`.  Reporting 'unknown' there leaves a
build unable to answer "what code produced this" from inside itself, which is
the same question the parameterization records answer for molecules.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import subprocess

import pytest

import htpolynet.external.software as software


@pytest.fixture(autouse=True)
def restore_git_commit():
    """software.git_commit is module state; put it back afterwards."""
    original = software.git_commit
    yield
    software.git_commit = original


def _no_git(monkeypatch):
    """Makes the git lookup fail the way it does outside a checkout."""
    def boom(*a, **kw):
        raise FileNotFoundError('git')
    monkeypatch.setattr(subprocess, 'run', boom)


class TestGitCommitReporting:

    def test_a_checkout_reports_a_commit(self):
        software.git_commit = 'unset'
        software._get_git_commit()
        # The test suite runs inside the repo, so this is a real lookup.
        assert software.git_commit != 'unset'
        assert 'unknown' not in software.git_commit

    def test_an_installed_copy_reports_its_version(self, monkeypatch):
        _no_git(monkeypatch)
        software.git_commit = 'unset'
        software._get_git_commit()
        assert 'installed version' in software.git_commit
        assert 'not a git checkout' in software.git_commit

    def test_an_installed_copy_never_reports_unknown(self, monkeypatch):
        # The container case 8a hit: htpolynet info said 'unknown', so a build
        # could not identify its own code.
        _no_git(monkeypatch)
        software.git_commit = 'unknown'
        software._get_git_commit()
        assert software.git_commit != 'unknown'

    def test_the_lookup_never_raises(self, monkeypatch):
        # Provenance reporting must not be able to abort a build.
        _no_git(monkeypatch)
        monkeypatch.setattr(software, '__name__', software.__name__)
        software._get_git_commit()
