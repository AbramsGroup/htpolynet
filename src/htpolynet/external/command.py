"""Thin wrapper around subprocess for running external commands.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import logging
import subprocess
import time

from .. import profiling

logger = logging.getLogger(__name__)

_LINELEN = 55


def opts(**kwargs):
    """Format keyword arguments as '-key value' flag pairs.

    Useful for building command strings from option dicts, e.g.::

        opts(f='in.gro', o='out.gro')  ->  '-f in.gro -o out.gro'

    Args:
        **kwargs: option name/value pairs

    Returns:
        str: space-separated '-key value' string
    """
    return ' '.join(f'-{k} {v}' for k, v in kwargs.items())


def run(command, ignore_codes=(), override=None, quiet=True):
    """Run a shell command and return its output.

    Args:
        command (str): shell command to run
        ignore_codes (tuple | list): return codes to treat as success, defaults to ()
        override (tuple[str, str] | None): ``(needle, msg)`` — if needle appears
            in stdout or stderr, raise even when returncode == 0, defaults to None
        quiet (bool): if False, log the command before running, defaults to True

    Returns:
        tuple[str, str]: (stdout, stderr)

    Raises:
        subprocess.SubprocessError: on non-zero return code (unless in ignore_codes),
            or when the override needle is found in output
    """
    if not quiet:
        logger.debug(command)
    _t0 = time.monotonic()
    process = subprocess.Popen(
        command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    out, err = process.communicate()
    rc = process.returncode
    profiling.record_subprocess(profiling.classify_command(command), time.monotonic() - _t0)

    def _log_output():
        if out:
            logger.error('stdout:\n' + '*' * _LINELEN + '\n' + out + '*' * _LINELEN)
        if err:
            logger.error('stderr:\n' + '*' * _LINELEN + '\n' + err + '*' * _LINELEN)

    if rc != 0 and rc not in ignore_codes:
        logger.error(f'Returncode: {rc}')
        _log_output()
        raise subprocess.SubprocessError(f'Command "{command}" failed with returncode {rc}')

    if override is not None:
        needle, msg = override
        if needle in out or needle in err:
            logger.error(f'Returncode: {rc}, but error detected: {msg}')
            _log_output()
            raise subprocess.SubprocessError(f'Command "{command}" triggered override: {msg}')

    return out, err