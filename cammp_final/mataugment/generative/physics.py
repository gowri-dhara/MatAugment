# mataugment/generative/physics.py
import torch
import torch.nn.functional as F
import numpy as np

# Micrograph physics
def fft_magnitude(img):
    # img: (B,1,H,W) in [-1,1]; convert to [0,1] for FFT stability
    x = (img + 1) / 2.0
    x = x.squeeze(1)  # (B,H,W)
    fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    mag = torch.abs(fft)
    return mag

def radial_profile(mag):
    # crude radial average for isotropy (expects square H==W)
    B, H, W = mag.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=mag.device),
                            torch.arange(W, device=mag.device), indexing='ij')
    r = torch.sqrt((xx - W//2)**2 + (yy - H//2)**2)
    r = r.long().clamp(max=min(H, W))
    nbins = r.max().item() + 1
    prof = []
    for b in range(B):
        sums = torch.zeros(nbins, device=mag.device)
        counts = torch.zeros(nbins, device=mag.device)
        sums.index_add_(0, r.view(-1), mag[b].reshape(-1))
        counts.index_add_(0, r.view(-1), torch.ones_like(mag[b]).reshape(-1))
        prof.append(sums / (counts + 1e-8))
    return torch.stack(prof, dim=0)  # (B, nbins)

def loss_power_spectrum(gen_imgs, ref_imgs):
    return F.l1_loss(fft_magnitude(gen_imgs), fft_magnitude(ref_imgs))

def loss_isotropy(gen_imgs):
    rp = radial_profile(fft_magnitude(gen_imgs))
    # isotropy -> low variance across radii
    return rp.var(dim=1).mean()

def loss_total_variation(imgs):
    # encourage smoothness; avoids checkerboard artifacts
    dh = (imgs[:, :, 1:, :] - imgs[:, :, :-1, :]).abs().mean()
    dw = (imgs[:, :, :, 1:] - imgs[:, :, :, :-1]).abs().mean()
    return dh + dw

# DFT point-cloud physics
def pairwise_distances(coords):
    # coords: (B,N,3) in [0,1]; unit cell assumed cubic here for simplicity
    B, N, _ = coords.shape
    diffs = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B,N,N,3)
    d = torch.sqrt((diffs**2).sum(dim=-1) + 1e-12)     # (B,N,N)
    return d

def loss_min_interatomic_distance(coords, min_d=0.05):
    d = pairwise_distances(coords)
    mask = ~torch.eye(d.size(1), dtype=torch.bool, device=d.device)
    d = d[:, mask].view(coords.size(0), -1)
    penalty = F.relu(min_d - d).mean()
    return penalty

def radial_distribution(coords, nbins=32, rmax=0.8):
    d = pairwise_distances(coords)  # (B,N,N)
    B, N, _ = d.shape
    mask = ~torch.eye(N, dtype=torch.bool, device=d.device)
    d = d[:, mask].view(B, -1)
    bins = torch.linspace(0, rmax, nbins+1, device=coords.device)
    hists = []
    for b in range(B):
        hist = torch.histc(d[b], bins=nbins, min=0, max=rmax)
        hists.append(hist / (hist.sum() + 1e-8))
    return torch.stack(hists, dim=0)

def loss_rdf_match(gen_coords, ref_coords, nbins=32, rmax=0.8):
    g_r_gen = radial_distribution(gen_coords, nbins, rmax)
    g_r_ref = radial_distribution(ref_coords, nbins, rmax)
    return F.mse_loss(g_r_gen, g_r_ref)

# Wrapper 
class PhysLoss:
    """
    Toggle physics terms with weights.
    mode='image' uses (power spectrum, isotropy, TV).
    mode='dft'   uses (RDF match, min-distance).
    """
    def __init__(self, mode='image',
                 w_power=1.0, w_iso=0.1, w_tv=0.05,
                 w_rdf=1.0, w_min_d=0.5, min_d=0.05):
        self.mode = mode
        self.w_power = w_power
        self.w_iso = w_iso
        self.w_tv = w_tv
        self.w_rdf = w_rdf
        self.w_min_d = w_min_d
        self.min_d = min_d

    def image_loss(self, gen_imgs, ref_imgs):
        lp = loss_power_spectrum(gen_imgs, ref_imgs) * self.w_power
        li = loss_isotropy(gen_imgs) * self.w_iso
        ltv = loss_total_variation(gen_imgs) * self.w_tv
        return lp + li + ltv, {'power': lp.item(), 'isotropy': li.item(), 'tv': ltv.item()}

    def dft_loss(self, gen_coords, ref_coords):
        lrdf = loss_rdf_match(gen_coords, ref_coords) * self.w_rdf
        lmin = loss_min_interatomic_distance(gen_coords, self.min_d) * self.w_min_d
        return lrdf + lmin, {'rdf': lrdf.item(), 'min_d': lmin.item()}

    def __call__(self, generated, reference):
        if self.mode == 'image':
            return self.image_loss(generated, reference)
        else:
            return self.dft_loss(generated, reference)
