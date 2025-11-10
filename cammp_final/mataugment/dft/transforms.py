'''

Transforms for DFT atomic structures.

This includes operations for rotation, perturbation, and adding noise to energies and forces which is useful for data augmentation
and robustness testing of DFT models.

'''

import numpy as np
from ase import Atoms
from ase.build import rotate  


# Rotation Transform
class RotateStructure:
    '''
    Rotate an atomic structure by a fixed or random angle about a given axis.
    It is useful for ensuring rotational invariance in models.
    '''

    def __init__(self, axis='z', angle=None):
        '''
        axis : str or np.ndarray
            Axis of rotation ('x', 'y', 'z' or a vector).
        angle : float or None
            Angle in degrees. If None, a random angle in [0, 360) is used.
        '''
        self.axis = axis
        self.angle = angle

    def __call__(self, atoms: Atoms):
        #Apply rotation to ASE Atoms object.
        if self.angle is None:
            angle = np.random.uniform(0, 360)
        else:
            angle = self.angle

        if isinstance(self.axis, str):
            axis_map = {'x': np.array([1, 0, 0]),
                        'y': np.array([0, 1, 0]),
                        'z': np.array([0, 0, 1])}
            axis_vec = axis_map.get(self.axis.lower(), np.array([0, 0, 1]))
        else:
            axis_vec = np.asarray(self.axis, dtype=float)

        atoms_copy = atoms.copy()
        atoms_copy.rotate(v=axis_vec, a=angle, rotate_cell=False)
        return atoms_copy


# Atomic Perturbation Transform
class PerturbAtoms:
    '''
    Randomly perturb atomic positions to simulate small displacements or thermal vibrations.
    '''

    def __init__(self, sigma=0.02):
        """
        sigma : float
            Standard deviation of Gaussian noise (in Å).
        """
        self.sigma = sigma

    def __call__(self, atoms: Atoms):
        atoms_copy = atoms.copy()
        perturbations = np.random.normal(0, self.sigma, atoms_copy.positions.shape)
        atoms_copy.positions += perturbations
        return atoms_copy


# Energy Noise Transform

class AddEnergyNoise:
    '''
    Add Gaussian noise to scalar energy values.
    '''

    def __init__(self, std=0.001):
        self.std = std

    def __call__(self, energy):
        noise = np.random.normal(0, self.std)
        return energy + noise


# Force Noise Transform

class AddForceNoise:
    '''
    Add Gaussian noise to atomic forces (Nx3 array).
    '''

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, forces):
        noise = np.random.normal(0, self.std, forces.shape)
        return forces + noise
