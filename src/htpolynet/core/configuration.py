"""Manages reading and parsing of YAML configuration files.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import json
import logging
import os

import yaml

logger = logging.getLogger(__name__)


class Configuration:
    """Pure data container for a build configuration file.

    Reads and validates a YAML or JSON file and exposes each top-level
    section as a named attribute.  Does NOT create Molecule or Reaction
    objects; that is the responsibility of Runtime.
    """

    def __init__(self):
        self.cfgfile = ''
        self.title = ''
        self.ncpu = os.cpu_count()
        self.constituents = {}       # {name: {count: N, ...}}
        self.reaction_specs = []     # raw reaction dicts
        self.initial_composition = []  # [{'molecule': name, 'count': N}]
        self.gromacs = {}
        self.densification = {}
        self.precure = {}
        self.postcure = {}
        self.cure = {}
        self.gaff = {}
        self.ambertools = {}
        self.resolve_type_discrepancies = []
        self.basedict = {}

    @classmethod
    def read(cls, filename):
        """Reads a JSON or YAML configuration file and returns a populated Configuration.

        Args:
            filename (str): path to configuration file

        Raises:
            Exception: if the file extension is not .json, .yaml, or .yml

        Returns:
            Configuration: populated configuration object
        """
        _, ext = os.path.splitext(filename)
        inst = cls()
        inst.cfgfile = filename
        if ext == '.json':
            with open(filename, 'r') as f:
                inst.basedict = json.load(f)
        elif ext in ('.yaml', '.yml'):
            with open(filename, 'r') as f:
                inst.basedict = yaml.safe_load(f)
        else:
            raise Exception(f'Unknown config file extension {ext}')
        inst._parse()
        return inst

    def _parse(self):
        """Populates named attributes from self.basedict."""
        self.title = self.basedict.get('Title', 'No title provided')
        self.ncpu = self.basedict.get('ncpu', os.cpu_count())
        self.constituents = self.basedict.get('constituents', {})
        self.reaction_specs = self.basedict.get('reactions', [])
        self.gromacs = self.basedict.get('gromacs', {})
        self.densification = self.basedict.get('densification', {})
        self.precure = self.basedict.get('precure', {})
        self.postcure = self.basedict.get('postcure', {})
        self.cure = self.basedict.get('CURE', {})
        self.gaff = self.basedict.get('GAFF', {})
        self.ambertools = self.basedict.get('ambertools', {})
        self.resolve_type_discrepancies = self.basedict.get('resolve_type_discrepancies', [])
        self.initial_composition = [
            {'molecule': mol, 'count': rec.get('count', 0)}
            for mol, rec in self.constituents.items()
        ]
