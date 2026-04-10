"""Crystal lattice generators.

Author: Cameron F. Abrams <cfa22@drexel.edu>
"""
import numpy as np
from itertools import product


def fcc(a, nc=(1, 1, 1)):
    """Generate atom positions for an FCC crystal supercell.

    Args:
        a (float): lattice parameter (same units as desired output)
        nc (sequence of int): number of unit cells along each lattice vector, defaults to (1,1,1)

    Returns:
        numpy.ndarray, shape (N, 3): fractional × a positions of all atoms
    """
    basis = np.identity(3) * a
    base_atoms = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]) * a
    positions = []
    for i, j, k in product(*[range(n) for n in nc]):
        origin = np.dot(basis, np.array([i, j, k]))
        for atom in base_atoms:
            positions.append(origin + atom)
    return np.array(positions)
