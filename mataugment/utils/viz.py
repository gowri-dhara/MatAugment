# path: mataugment/utils/viz.py
import numpy as np
import matplotlib.pyplot as plt
import os

def show_image_grid(pair_tuple, ncol=4, fname=None):
    """
    pair_tuple = (original_batch, reconstructed_batch)
    each is torch.Tensor (B,1,H,W) normalized to [-1,1]
    """
    orig, rec = pair_tuple
    B = min(orig.shape[0], ncol)
    fig, axs = plt.subplots(2, B, figsize=(3*B, 6))
    for i in range(B):
        o = ((orig[i].numpy().transpose(1,2,0) + 1)*127.5).astype(np.uint8)[:,:,0]
        r = ((rec[i].numpy().transpose(1,2,0) + 1)*127.5).astype(np.uint8)[:,:,0]
        axs[0,i].imshow(o, cmap='gray'); axs[0,i].axis('off'); axs[0,i].set_title("orig")
        axs[1,i].imshow(r, cmap='gray'); axs[1,i].axis('off'); axs[1,i].set_title("recon")
    plt.tight_layout()
    if fname:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname)
    else:
        plt.show()
    plt.close(fig)

def visualize_pointcloud_numpy(coords, fname=None):
    """
    coords: (N,3) numpy array; simple 3d scatter saved to PNG
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=20)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.tight_layout()
    if fname:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname)
    else:
        plt.show()
    plt.close(fig)
