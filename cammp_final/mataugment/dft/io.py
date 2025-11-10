# mataugment/dft/io.py
import os
from ase.io import read

def load_all_structures(folder, ext=".xyz"):
    #Loads all structures with a given extension from a folder.
    structures = []
    for f in os.listdir(folder):
        if f.endswith(ext):
            structures.append(read(os.path.join(folder, f)))
    return structures
