"""Shared logging configuration for HTPolyNet subcommands.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import logging
import os
import shutil

from ..utils.banner import banner

_logger = logging.getLogger(__name__)


def setup_logging(loglevel, diag=None, no_banner=False):
    """Configures the root logger with optional file and console handlers.

    Args:
        loglevel (str): log level name for the diagnostic file (e.g. 'debug', 'info')
        diag (str): path to the diagnostic log file; if None, file logging is skipped
        no_banner (bool): if True, suppresses the HTPolyNet startup banner
    """
    loglevel_numeric = getattr(logging, loglevel.upper())
    if diag:
        # File-only basicConfig; then add a console handler so INFO-level
        # messages also reach the terminal.
        if os.path.exists(diag):
            n = 1
            while os.path.exists(f'#{n}#{diag}'):
                n += 1
            shutil.copyfile(diag, f'#{n}#{diag}')
        logging.basicConfig(
            filename=diag, filemode='w',
            format='%(asctime)s %(name)s.%(funcName)s %(levelname)s> %(message)s',
            level=loglevel_numeric,
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(levelname)s> %(message)s'))
        logging.getLogger('').addHandler(console)
    else:
        # Console-only — basicConfig already installs a StreamHandler on
        # the root logger by default.  Adding another would emit every
        # record twice (this was the bug behind double-printed INFO lines
        # under `htpolynet plots`, `analyze`, `postsim`).
        logging.basicConfig(
            format='%(levelname)s> %(message)s',
            level=loglevel_numeric,
        )
    if not no_banner:
        banner(_logger.info)
