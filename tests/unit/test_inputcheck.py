"""Tests the ``htpolynet input-check`` subcommand.

input-check is the documented pre-flight before committing a build to a
queue -- the container docs now tell users to size their core request from
its atom count -- so its arithmetic needs to be right.  It had no coverage.

``input_check`` resolves ``./lib/molecules`` relative to the working
directory, so every test here runs in its own tmp_path.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import importlib.resources
import os
import shutil
from argparse import Namespace

import pytest

from htpolynet.utils.inputcheck import input_check


def _resource(name):
    return str(importlib.resources.files('htpolynet.resources').joinpath(f'molecules/inputs/{name}'))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A cwd containing lib/molecules/inputs, as input_check expects."""
    (tmp_path / 'lib' / 'molecules' / 'inputs').mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _seed(workdir, mol2_name):
    shutil.copy(_resource(mol2_name), workdir / 'lib' / 'molecules' / 'inputs' / mol2_name)


def _config(workdir, body):
    p = workdir / 'cfg.yaml'
    p.write_text(body)
    return Namespace(config=str(p))


def _n_atoms(mol2_name):
    from htpolynet.core.coordinates import Coordinates
    return Coordinates.read_mol2(_resource(mol2_name)).A.shape[0]


def test_counts_atoms_for_a_single_constituent(workdir, capsys):
    _seed(workdir, 'STY.mol2')
    args = _config(workdir, 'Title: t\nconstituents:\n  STY:\n    count: 10\n')
    input_check(args)
    out = capsys.readouterr().out
    n = _n_atoms('STY.mol2')
    assert f'Molecule STY: {n} atoms, 10 molecules' in out
    assert f'{n * 10} atoms in initial system' in out


def test_sums_atoms_across_constituents(workdir, capsys):
    _seed(workdir, 'STY.mol2')
    _seed(workdir, 'GMA.mol2')
    args = _config(workdir,
                   'Title: t\nconstituents:\n  STY:\n    count: 10\n  GMA:\n    count: 4\n')
    input_check(args)
    out = capsys.readouterr().out
    expected = 10 * _n_atoms('STY.mol2') + 4 * _n_atoms('GMA.mol2')
    assert f'{expected} atoms in initial system' in out


def test_zero_count_constituent_contributes_nothing(workdir, capsys):
    """A count of 0 is how the depot declares cap templates that never enter the box."""
    _seed(workdir, 'STY.mol2')
    _seed(workdir, 'GMA.mol2')
    args = _config(workdir,
                   'Title: t\nconstituents:\n  STY:\n    count: 10\n  GMA:\n    count: 0\n')
    input_check(args)
    out = capsys.readouterr().out
    assert 'Molecule GMA' not in out
    assert f'{10 * _n_atoms("STY.mol2")} atoms in initial system' in out


def test_constituent_with_no_structure_file_is_not_counted(workdir, capsys):
    """Nothing on disk for NOPE: it must be skipped, not crash or count as zero-atom."""
    _seed(workdir, 'STY.mol2')
    args = _config(workdir,
                   'Title: t\nconstituents:\n  STY:\n    count: 2\n  NOPE:\n    count: 5\n')
    input_check(args)
    out = capsys.readouterr().out
    assert 'Molecule NOPE' not in out
    assert f'{2 * _n_atoms("STY.mol2")} atoms in initial system' in out


def test_empty_constituents_reports_zero(workdir, capsys):
    args = _config(workdir, 'Title: t\nconstituents: {}\n')
    input_check(args)
    assert '0 atoms in initial system' in capsys.readouterr().out


def test_no_weight_percent_reported_without_parameterized_masses(workdir, capsys):
    """Masses come from .top files under lib/molecules/parameterized; with only
    mol2 inputs there is no mass, so the wt-% block must be suppressed."""
    _seed(workdir, 'STY.mol2')
    args = _config(workdir, 'Title: t\nconstituents:\n  STY:\n    count: 10\n')
    input_check(args)
    assert 'wt-%' not in capsys.readouterr().out


def test_names_the_config_file_in_its_summary(workdir, capsys):
    _seed(workdir, 'STY.mol2')
    args = _config(workdir, 'Title: t\nconstituents:\n  STY:\n    count: 1\n')
    input_check(args)
    assert os.path.basename(args.config) in capsys.readouterr().out
