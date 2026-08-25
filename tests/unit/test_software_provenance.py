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


@pytest.fixture(autouse=True)
def clear_baked_commit(monkeypatch):
    """The baked commit is environment state; a developer machine has none."""
    monkeypatch.delenv('HTPOLYNET_COMMIT', raising=False)


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


class TestBakedCommit:
    """The published container has no git and no .git beside the package, and
    its version string is misleading between releases -- the scheduled rebuild
    builds from main HEAD and reports whatever pyproject.toml last said.  The
    image therefore bakes its commit in at build time.
    """

    def test_baked_commit_is_reported_when_git_is_unavailable(self, monkeypatch):
        _no_git(monkeypatch)
        monkeypatch.setenv('HTPOLYNET_COMMIT', 'a1b2c3d4e5f6a7b8c9d0')
        software.git_commit = 'unset'
        software._get_git_commit()
        assert 'a1b2c3d' in software.git_commit
        assert 'image build' in software.git_commit

    def test_baked_commit_beats_the_version_fallback(self, monkeypatch):
        # The whole point: a version string alone cannot distinguish a release
        # image from a weekly rebuild of untagged main.
        _no_git(monkeypatch)
        monkeypatch.setenv('HTPOLYNET_COMMIT', 'a1b2c3d4e5f6a7b8c9d0')
        software.git_commit = 'unset'
        software._get_git_commit()
        assert 'installed version' not in software.git_commit

    def test_the_dockerfile_default_is_not_reported_as_a_commit(self, monkeypatch):
        # ARG HTPOLYNET_COMMIT=unknown is the default for a local docker build
        # with no --build-arg; it must not be echoed back as if it were a sha.
        _no_git(monkeypatch)
        monkeypatch.setenv('HTPOLYNET_COMMIT', 'unknown')
        software.git_commit = 'unset'
        software._get_git_commit()
        assert 'installed version' in software.git_commit

    def test_an_empty_value_falls_through(self, monkeypatch):
        _no_git(monkeypatch)
        monkeypatch.setenv('HTPOLYNET_COMMIT', '   ')
        software.git_commit = 'unset'
        software._get_git_commit()
        assert 'installed version' in software.git_commit

    def test_a_real_checkout_still_wins(self, monkeypatch):
        # In a dev checkout git is more authoritative: it reflects the working
        # tree, including uncommitted changes, which a baked value cannot.
        monkeypatch.setenv('HTPOLYNET_COMMIT', 'a1b2c3d4e5f6a7b8c9d0')
        software.git_commit = 'unset'
        software._get_git_commit()
        assert 'image build' not in software.git_commit
