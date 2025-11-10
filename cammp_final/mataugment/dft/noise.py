# mataugment/dft/noise.py
import numpy as np

class AddEnergyNoise:
    '''
    Add Gaussian noise to total energy values (purpose: to simulate DFT error or precision limits).
    '''
    def __init__(self, std=0.005):
        self.std = std  # eV

    def __call__(self, energy):
        noise = np.random.normal(0, self.std)
        return energy + noise


class AddForceNoise:
    '''
    Add Gaussian noise to atomic forces to simulate DFT-level uncertainty.
    '''
    def __init__(self, std=0.01):
        self.std = std  # eV/Å

    def __call__(self, forces):
        noisy = forces + np.random.normal(0, self.std, size=forces.shape)
        return noisy
