"""Generates SLURM submission scripts for HTPolyNet runs.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""

# #SBATCH directives emitted in this fixed order; any other key in the slurm
# config dict is passed through verbatim with underscores mapped to hyphens.
_KNOWN_DIRECTIVES = [
    ('job_name',        'job-name'),
    ('partition',       'partition'),
    ('account',         'account'),
    ('nodes',           'nodes'),
    ('ntasks',          'ntasks'),
    ('ntasks_per_node', 'ntasks-per-node'),
    ('cpus_per_task',   'cpus-per-task'),
    ('time',            'time'),
    ('mem',             'mem'),
    ('mem_per_cpu',     'mem-per-cpu'),
    ('gres',            'gres'),
    ('constraint',      'constraint'),
    ('output',          'output'),
    ('error',           'error'),
]
_KNOWN_KEYS = {k for k, _ in _KNOWN_DIRECTIVES} | {'preamble', 'sif', 'apptainer_alias'}


def _normalize_keys(d):
    """Returns a copy of dict d with all keys lowercased and hyphens → underscores."""
    return {k.lower().replace('-', '_'): v for k, v in d.items()}


def generate_script(config_file, slurm_cfg, run_args,
                    native=False, sif=None, apptainer_alias='apptainer'):
    """Generates the text of a SLURM submission script.

    Args:
        config_file (str): path to the htpolynet YAML config file
        slurm_cfg (dict): slurm section from the config (already normalized)
        run_args (dict): htpolynet run options (proj, loglevel, flags …)
        native (bool): run htpolynet natively; otherwise via container
        sif (str): path to the Apptainer/Singularity .sif image
        apptainer_alias (str): container runtime command name

    Returns:
        str: complete script text
    """
    lines = ['#!/bin/bash']

    # --- known #SBATCH directives in canonical order ---
    for key, flag in _KNOWN_DIRECTIVES:
        val = slurm_cfg.get(key)
        if val is not None:
            lines.append(f'#SBATCH --{flag}={val}')

    # --- pass-through directives (unrecognised keys) ---
    for key, val in slurm_cfg.items():
        if key not in _KNOWN_KEYS:
            lines.append(f'#SBATCH --{key.replace("_", "-")}={val}')

    lines.append('')

    if native:
        # --- optional preamble (module loads, conda activate, …) ---
        preamble = slurm_cfg.get('preamble', [])
        for line in preamble:
            lines.append(line)
        if preamble:
            lines.append('')
        lines.append(_build_run_cmd(config_file, run_args))
    else:
        use_gpu = 'gpu' in str(slurm_cfg.get('gres', ''))
        nv = ' --nv' if use_gpu else ''
        lines.append(f'{apptainer_alias} exec{nv} \\')
        lines.append(f'    --bind $(pwd):$(pwd) --pwd $(pwd) \\')
        lines.append(f'    {sif} \\')
        lines.append(f'    {_build_run_cmd(config_file, run_args)}')

    lines.append('')
    return '\n'.join(lines)


def _build_run_cmd(config_file, run_args):
    """Builds the 'htpolynet run …' command string from run_args dict.

    Args:
        config_file (str): config file path
        run_args (dict): keys mirror the htpolynet run CLI options

    Returns:
        str: the full command string
    """
    parts = ['htpolynet', 'run', config_file]
    if run_args.get('lib', 'lib') != 'lib':
        parts += ['-lib', run_args['lib']]
    if run_args.get('proj', 'next') != 'next':
        parts += ['-proj', run_args['proj']]
    if run_args.get('diag', 'htpolynet_runtime_diagnostics.log') != 'htpolynet_runtime_diagnostics.log':
        parts += ['-diag', run_args['diag']]
    if run_args.get('restart'):
        parts.append('-restart')
    if run_args.get('loglevel', 'info') != 'info':
        parts += ['--loglevel', run_args['loglevel']]
    if run_args.get('force_parameterization'):
        parts.append('--force-parameterization')
    if run_args.get('force_checkin'):
        parts.append('--force-checkin')
    if run_args.get('param_only'):
        parts.append('--param-only')
    if run_args.get('no_banner'):
        parts.append('--no-banner')
    return ' '.join(parts)
