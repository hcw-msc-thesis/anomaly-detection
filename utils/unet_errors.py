import torch

from models.unet import ssim_map


def unet_error(target, recon, log_b, uncertainty_pct, error):
    if error in ("l1", "unc_l1"):
         diff = l1_error(target, recon)
    elif error in ("ssim", "unc_ssim"):
         diff = ssim_error(target, recon)
    elif error in ("chroma", "unc_chroma"):
         diff = chroma_error(target, recon)
    else:
         return None
    b = log_b.exp().clamp(min=max(uncertainty_pct, 1e-6))

    if not error.startswith("unc_"):
         return diff

    if error == "unc_chroma":
         b = b.mean(1, keepdim=True).expand_as(b)

    return diff / b


def chroma_error(target, recon):
    def cbcr(img):
            #Renormalise from [-1, 1] to [0, 1]
            r, g, b = ((img + 1) / 2).unbind(1)  
            y = 0.299 * r + 0.587 * g + 0.114 * b
            return torch.stack([0.564 * (b - y), 0.713 * (r - y)], 1)
    return (cbcr(recon) - cbcr(target)).abs().sum(1, keepdim=True).expand(-1, 3, -1, -1)

def ssim_error(target, recon):
    return (1 - ssim_map(target, recon)).clamp(min=0)

def l1_error(target, recon):
    return (recon - target).abs()
