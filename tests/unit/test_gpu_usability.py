"""Tests for GPU-usability detection in htpolynet.external.software.

The interesting case is a gmx binary whose GPU support is neither 'CUDA'
nor 'disabled': conda-forge (and hence our container image) ships an
OpenCL build, which cannot drive the NVIDIA devices nvidia-smi reports.
"""
import pytest

from htpolynet.external import software as sw


@pytest.fixture
def gpu_state(monkeypatch):
    """Sets module-level gpu_ids and the parsed gmx GPU backend."""
    def _set(gpu_ids, backend):
        monkeypatch.setattr(sw, 'gpu_ids', gpu_ids)
        monkeypatch.setattr(sw, 'versions', dict(sw.versions, gromacs_gpu=backend))
    return _set


def test_cuda_build_with_gpus_is_usable(gpu_state):
    gpu_state([0], 'CUDA')
    assert sw.gpu_unusable_reasons() == []


def test_no_gpus_detected(gpu_state):
    gpu_state([], 'CUDA')
    reasons = sw.gpu_unusable_reasons()
    assert len(reasons) == 1
    assert 'no GPU devices detected' in reasons[0]


def test_gmx_built_without_gpu_support(gpu_state):
    gpu_state([0], 'disabled')
    reasons = sw.gpu_unusable_reasons()
    assert len(reasons) == 1
    assert 'disabled' in reasons[0]


def test_opencl_build_cannot_drive_nvidia_devices(gpu_state):
    """The container case: OpenCL is not 'disabled', but is still unusable."""
    gpu_state([0], 'OpenCL')
    reasons = sw.gpu_unusable_reasons()
    assert len(reasons) == 1
    assert 'OpenCL' in reasons[0]


def test_opencl_build_with_no_gpus_reports_only_missing_hardware(gpu_state):
    gpu_state([], 'OpenCL')
    reasons = sw.gpu_unusable_reasons()
    assert len(reasons) == 1
    assert 'no GPU devices detected' in reasons[0]


def test_mdrun_cmd_forces_cpu_when_unusable(gpu_state):
    gpu_state([0], 'OpenCL')
    assert sw._mdrun_cmd('gmx mdrun') == 'gmx mdrun -nb cpu'


def test_mdrun_cmd_leaves_command_alone_when_usable(gpu_state):
    gpu_state([0], 'CUDA')
    assert sw._mdrun_cmd('gmx mdrun') == 'gmx mdrun'


def test_enforce_strips_gpu_id_for_opencl_build(gpu_state):
    gpu_state([0], 'OpenCL')
    cfg = {'mdrun_options': {'gpu_id': 0, 'ntomp': 4}}
    sw._enforce_gpu_consistency(cfg)
    assert 'gpu_id' not in cfg['mdrun_options']
    assert cfg['mdrun_options']['ntomp'] == 4


def test_enforce_keeps_gpu_id_for_cuda_build(gpu_state):
    gpu_state([0], 'CUDA')
    cfg = {'mdrun_options': {'gpu_id': 0}}
    sw._enforce_gpu_consistency(cfg)
    assert cfg['mdrun_options']['gpu_id'] == 0
