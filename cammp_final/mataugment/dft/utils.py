# mataugment/dft/utils.py
from ase.io import read, write
from ase.visualize import view
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def load_structure(path):
    #Load an atomic structure using ASE (supports .xyz, .vasp, .cif, etc.)
    return read(path)

def save_structure(structure, path):
    #Save an ASE structure to file.
    write(path, structure)

def visualize_structure(structure):
    #Open ASE viewer for quick visualization.
    view(structure)


def visualize_structure_3d(structure, title="Atomic Structure"):
    '''
    Visualize an ASE Atoms object in 3D.
    '''
    coords = structure.get_positions()
    cell = structure.get_cell_lengths_and_angles()[:3]
    coords = coords / cell  # normalize to [0,1]
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               s=40, c='steelblue', edgecolor='k', alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    plt.show()


def visualize_pointcloud(coords, title="Generated Structure"):
    '''
    Visualize generated coordinates (Tensor or NumPy array of shape (N, 3)).
    '''
    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    coords = np.clip(coords, 0, 1)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               s=40, c='tomato', edgecolor='k', alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    plt.show()


def compare_real_and_generated_3d(real_cif_path, generated_coords, title="Real vs Generated Structure"):
    '''
    Compare a real CIF structure (from file) and a generated point cloud.
    Shows them side-by-side in 3D scatter plots.
    '''
    # Load real structure 
    real_atoms = read(real_cif_path)
    real_coords = real_atoms.get_positions()
    cell = real_atoms.get_cell_lengths_and_angles()[:3]
    real_coords = real_coords / cell  # normalize to [0,1]
    real_coords = np.clip(real_coords, 0, 1)

    # Prepare generated structure
    if hasattr(generated_coords, "detach"):
        generated_coords = generated_coords.detach().cpu().numpy()
    generated_coords = np.clip(generated_coords, 0, 1)

    # Plot 
    fig = plt.figure(figsize=(10, 5))

    # Real structure
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(real_coords[:, 0], real_coords[:, 1], real_coords[:, 2],
                s=40, c='steelblue', edgecolor='k', alpha=0.8)
    ax1.set_title("Real Structure")
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # Generated structure
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(generated_coords[:, 0], generated_coords[:, 1], generated_coords[:, 2],
                s=40, c='tomato', edgecolor='k', alpha=0.8)
    ax2.set_title("Generated Structure")
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def compare_structures_quantitatively(real_cif_path, generated_coords, bins=30, plot=True):
   '''
    Quantitatively compare a real CIF structure and a generated point cloud.
    Computes:
      - RMSD (root mean square deviation)
      - Pairwise distance histogram similarity (L2 distance)
      - Optional radial distribution plots
   '''
    # Load real structure 
    real_atoms = read(real_cif_path)
    real_coords = real_atoms.get_positions()
    cell = real_atoms.get_cell_lengths_and_angles()[:3]
    real_coords = real_coords / cell
    real_coords = np.clip(real_coords, 0, 1)

    # Prepare generated coords 
    if hasattr(generated_coords, "detach"):
        generated_coords = generated_coords.detach().cpu().numpy()
    generated_coords = np.clip(generated_coords, 0, 1)

    # Match number of atoms 
    n = min(len(real_coords), len(generated_coords))
    real_coords = real_coords[:n]
    generated_coords = generated_coords[:n]

    # RMSD
    rmsd = np.sqrt(np.mean((real_coords - generated_coords)**2))

    # Pairwise distances 
    real_d = cdist(real_coords, real_coords)
    gen_d = cdist(generated_coords, generated_coords)

    # Flatten upper triangles
    iu = np.triu_indices(n, k=1)
    real_flat = real_d[iu]
    gen_flat = gen_d[iu]

    # Normalize histograms
    hist_r, bins_r = np.histogram(real_flat, bins=bins, range=(0, 1), density=True)
    hist_g, bins_g = np.histogram(gen_flat, bins=bins, range=(0, 1), density=True)
    hist_diff = np.sqrt(np.mean((hist_r - hist_g)**2))

    # Optional plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(bins_r[:-1], hist_r, label="Real RDF", color='steelblue')
        plt.plot(bins_g[:-1], hist_g, label="Generated RDF", color='tomato')
        plt.title(f"RDF Comparison\nRMSD={rmsd:.4f} | Histogram Δ={hist_diff:.4f}")
        plt.xlabel("Distance (normalized)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "RMSD": float(rmsd),
        "HistogramDifference": float(hist_diff),
        "RealDistances": real_flat,
        "GeneratedDistances": gen_flat
    }
