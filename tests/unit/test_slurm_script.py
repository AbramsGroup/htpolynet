"""Tests SLURM batch-script generation.

``htpolynet gen-slurm-script`` is the documented way to launch a build on an
HPC cluster, and the container docs now lead with it -- but the module had no
test coverage at all.  These tests pin the emitted directives, their order,
and the difference between the containerized and native run lines.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import pytest

from htpolynet.external.slurm import generate_script, _build_run_cmd, _normalize_keys


SIF = '/shared/containers/htpolynet.sif'


def _lines(text):
    return [ln for ln in text.split('\n') if ln.strip()]


def _directives(text):
    return [ln for ln in text.split('\n') if ln.startswith('#SBATCH')]


# --- key normalization -----------------------------------------------------

def test_normalize_keys_lowercases_and_maps_hyphens():
    assert _normalize_keys({'Job-Name': 'x', 'CPUS-PER-TASK': 4}) == {
        'job_name': 'x', 'cpus_per_task': 4,
    }


def test_normalize_keys_leaves_conforming_keys_alone():
    assert _normalize_keys({'partition': 'def'}) == {'partition': 'def'}


# --- #SBATCH block ---------------------------------------------------------

def test_script_starts_with_a_shebang():
    out = generate_script('c.yaml', {}, {}, sif=SIF)
    assert out.split('\n')[0] == '#!/bin/bash'


def test_known_directives_are_emitted_in_canonical_order():
    """Order is fixed by _KNOWN_DIRECTIVES, not by dict insertion order."""
    cfg = {'time': '8:00:00', 'partition': 'def', 'job_name': 'ex6', 'nodes': 1}
    got = _directives(generate_script('c.yaml', cfg, {}, sif=SIF))
    assert got == [
        '#SBATCH --job-name=ex6',
        '#SBATCH --partition=def',
        '#SBATCH --nodes=1',
        '#SBATCH --time=8:00:00',
    ]


def test_absent_directives_are_omitted_entirely():
    out = generate_script('c.yaml', {'partition': 'def'}, {}, sif=SIF)
    assert _directives(out) == ['#SBATCH --partition=def']


def test_underscore_keys_become_hyphenated_flags():
    out = generate_script('c.yaml', {'cpus_per_task': 16, 'ntasks_per_node': 2}, {}, sif=SIF)
    assert '#SBATCH --cpus-per-task=16' in out
    assert '#SBATCH --ntasks-per-node=2' in out


def test_unrecognized_keys_pass_through_as_directives():
    out = generate_script('c.yaml', {'mail_type': 'END', 'exclusive': 'true'}, {}, sif=SIF)
    assert '#SBATCH --mail-type=END' in out
    assert '#SBATCH --exclusive=true' in out


def test_control_keys_are_not_emitted_as_directives():
    """sif/apptainer_alias/preamble configure generation; they are not SBATCH flags."""
    cfg = {'sif': SIF, 'apptainer_alias': 'singularity', 'preamble': ['module load x']}
    out = generate_script('c.yaml', cfg, {}, sif=SIF)
    assert not any('--sif' in d or '--apptainer-alias' in d or '--preamble' in d
                   for d in _directives(out))


# --- containerized invocation ---------------------------------------------

def test_container_mode_execs_through_apptainer():
    out = generate_script('c.yaml', {}, {}, sif=SIF)
    assert 'apptainer exec' in out
    assert SIF in out
    assert '--bind $(pwd):$(pwd) --pwd $(pwd)' in out
    assert out.rstrip().endswith('htpolynet run c.yaml')


def test_container_runtime_alias_is_honored():
    out = generate_script('c.yaml', {}, {}, sif=SIF, apptainer_alias='singularity')
    assert 'singularity exec' in out
    assert 'apptainer exec' not in out


def test_no_nv_flag_without_a_gpu_gres():
    out = generate_script('c.yaml', {'partition': 'def'}, {}, sif=SIF)
    assert '--nv' not in out


def test_nv_flag_added_for_a_gpu_gres():
    out = generate_script('c.yaml', {'gres': 'gpu:v100:1'}, {}, sif=SIF)
    assert 'exec --nv' in out


def test_non_gpu_gres_does_not_trigger_nv():
    out = generate_script('c.yaml', {'gres': 'lscratch:100'}, {}, sif=SIF)
    assert '--nv' not in out


# --- native invocation -----------------------------------------------------

def test_native_mode_has_no_container_invocation():
    out = generate_script('c.yaml', {}, {}, native=True)
    assert 'apptainer' not in out
    assert 'htpolynet run c.yaml' in out


def test_native_mode_emits_preamble_before_the_run_line():
    cfg = {'preamble': ['module purge', 'module load gromacs']}
    out = _lines(generate_script('c.yaml', cfg, {}, native=True))
    assert out.index('module purge') < out.index('module load gromacs')
    assert out.index('module load gromacs') < out.index('htpolynet run c.yaml')


def test_preamble_is_ignored_in_container_mode():
    cfg = {'preamble': ['module load gromacs']}
    assert 'module load gromacs' not in generate_script('c.yaml', cfg, {}, sif=SIF)


# --- the htpolynet run command --------------------------------------------

def test_run_cmd_defaults_are_left_implicit():
    """Defaults are omitted so the emitted line stays readable."""
    cmd = _build_run_cmd('c.yaml', {'lib': 'lib', 'proj': 'next', 'loglevel': 'info'})
    assert cmd == 'htpolynet run c.yaml'


def test_run_cmd_includes_non_default_values():
    cmd = _build_run_cmd('c.yaml', {'proj': 'proj-7', 'diag': 'diag.log', 'loglevel': 'debug'})
    assert '-proj proj-7' in cmd
    assert '-diag diag.log' in cmd
    assert '--loglevel debug' in cmd


@pytest.mark.parametrize('key,flag', [
    ('restart', '-restart'),
    ('force_parameterization', '--force-parameterization'),
    ('force_checkin', '--force-checkin'),
    ('param_only', '--param-only'),
    ('no_banner', '--no-banner'),
])
def test_run_cmd_boolean_flags(key, flag):
    assert flag in _build_run_cmd('c.yaml', {key: True})
    assert flag not in _build_run_cmd('c.yaml', {key: False})


def test_run_cmd_names_the_config_first():
    cmd = _build_run_cmd('my-config.yaml', {'proj': 'p'})
    assert cmd.startswith('htpolynet run my-config.yaml')
