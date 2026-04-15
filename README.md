# htpolynet
> High-Throughput Polymer Network Atomistic Simulations

htpolynet is a Python utility for generating atomistic models of cross-linked polymer networks together with appropriate topology and parameter files required for molecular dynamics simulations using Gromacs.  It is intended as a fully automated system builder requiring as inputs only the molecular structures of any monomer species, a description of the polymerization chemistry, and a handful of options describing desired system size and composition.  htpolynet uses the Generalized Amber Force Field for atom-typing and parameter generation.

## Installation

From PyPI:
```bash
pip install htpolynet
```

From conda-forge:
```bash
conda install -c conda-forge htpolynet
```

From source:
```bash
git clone git@github.com:AbramsGroup/htpolynet.git
cd htpolynet
pip install -e .
```

Once installed, the user has access to the main ``htpolynet`` command.

IMPORTANT NOTES: The programs ``antechamber``, ``parmchk2`` and ``tleap`` from AmberTools must be in your path.  These can be installed using the ``ambertools`` package from ``conda-forge`` or compiled from source.  You also need Gromacs installed so ``gmx`` is in your path.  The examples show how to build input monomer structures using OpenBabel, so to use them you need ``obabel`` in your path as well.

## Documentation

Please consult documentation at [abramsgroup.github.io/htpolynet](https://abramsgroup.github.io/htpolynet/).

## Meta

Cameron F. Abrams – cfa22@drexel.edu

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/cameronabrams](https://github.com/cameronabrams/)

[https://github.com/AbramsGroup](https://github.com/AbramsGroup/)

## Contributing

1. Fork it (<https://github.com/AbramsGroup/htolynet/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

