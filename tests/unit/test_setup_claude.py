"""Tests for the setup-claude subcommand, which installs the bundled skill.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import argparse as ap

from importlib.resources import files as pkg_files

from htpolynet.cli import setup_claude


def _args(skills_dir, force=False):
    return ap.Namespace(skills_dir=str(skills_dir), force=force)


def test_skill_ships_with_the_package():
    """The skill has to be package data, not a repo file.  Reaching an installed
    user is the whole point of the subcommand."""
    source = pkg_files('htpolynet.resources').joinpath('claude', 'SKILL.md')
    assert source.is_file()
    text = source.read_text(encoding='utf-8')
    assert text.startswith('---\n')
    assert 'name: htpolynet' in text


def test_skill_does_not_point_at_repo_paths():
    """A shipped skill cannot tell its reader to open a file in the source tree;
    an installed user has no docs/ directory.  This is what made the previous
    router-style skill unshippable."""
    text = pkg_files('htpolynet.resources').joinpath('claude', 'SKILL.md').read_text(encoding='utf-8')
    assert 'docs/source' not in text
    assert 'htpolynet.readthedocs.io' in text


def test_installs_into_the_named_directory(tmp_path):
    setup_claude(_args(tmp_path))
    target = tmp_path / 'htpolynet' / 'SKILL.md'
    assert target.is_file()
    source = pkg_files('htpolynet.resources').joinpath('claude', 'SKILL.md')
    assert target.read_text(encoding='utf-8') == source.read_text(encoding='utf-8')


def test_does_not_overwrite_without_force(tmp_path, capsys):
    target = tmp_path / 'htpolynet' / 'SKILL.md'
    target.parent.mkdir(parents=True)
    target.write_text('edited by the user', encoding='utf-8')
    setup_claude(_args(tmp_path))
    assert target.read_text(encoding='utf-8') == 'edited by the user'
    assert 'already exists' in capsys.readouterr().out


def test_force_overwrites(tmp_path):
    target = tmp_path / 'htpolynet' / 'SKILL.md'
    target.parent.mkdir(parents=True)
    target.write_text('stale', encoding='utf-8')
    setup_claude(_args(tmp_path, force=True))
    assert target.read_text(encoding='utf-8') != 'stale'


def test_expands_a_tilde_path(tmp_path, monkeypatch):
    """The default is ~/.claude/skills, so expansion is on the happy path."""
    monkeypatch.setenv('HOME', str(tmp_path))
    setup_claude(_args('~/.claude/skills'))
    assert (tmp_path / '.claude' / 'skills' / 'htpolynet' / 'SKILL.md').is_file()
