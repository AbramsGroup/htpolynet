"""Manages the inputcheck subcommand.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
from htpolynet.core.topocoord import TopoCoord
from htpolynet.core.configuration import Configuration
from htpolynet.core.coordinates import Coordinates
from htpolynet.external.command import Command
import os

def input_check(args):
    """Manages the input-check subcommand.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    lib='./lib/molecules'
    C=Configuration.read(args.config)
    natoms=0
    tmass=0.0
    mmass={}
    for mname,mrec in C.constituents.items():
        count=mrec.get('count',0)
        matoms=0
        mmass[mname]=0.0
        if os.path.exists(os.path.join(lib,'parameterized',f'{mname}.top')):
            TC=TopoCoord(grofilename=os.path.join(lib,'parameterized',f'{mname}.gro'),topfilename=os.path.join(lib,'parameterized',f'{mname}.top'))
            matoms=TC.Coordinates.A.shape[0]
            mmass[mname]=TC.Topology.total_mass()
        elif os.path.exists(os.path.join(lib,'inputs',f'{mname}.mol2')):
            c=Coordinates.read_mol2(os.path.join(lib,'inputs',f'{mname}.mol2'))
            matoms=c.A.shape[0]
        elif os.path.exists(os.path.join(lib,'inputs',f'{mname}.pdb')):
            out,err=Command(f'grep -c ^ATOM {os.path.join(lib,"inputs",f"{mname}.pdb")}').run(ignore_codes=[1])
            matoms=int(out)
            out,err=Command(f'grep -c ^HETATM {os.path.join(lib,"inputs",f"{mname}.pdb")}').run(ignore_codes=[1])
            matoms+=int(out)
        if matoms>0 and count:
            natoms+=count*matoms
            tmass+=count*mmass[mname]
            print(f'Molecule {mname}: {matoms} atoms, {count} molecules')
    print(f'{args.config}: {natoms} atoms in initial system.')
    if tmass>0.0:
        for mname,mrec in C.constituents.items():
            count=mrec.get('count',0)
            if count:
                wtpct=100.0*count*mmass[mname]/tmass
                print(f'Molecule {mname}: {wtpct:0.2f} wt-%')
