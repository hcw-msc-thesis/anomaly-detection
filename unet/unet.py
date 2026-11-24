"""
U-Net implementation

Much of this code is based off of the U-Net found in this Github: https://github.com/dome272/Diffusion-Models-pytorch
The code is used under the Apache 2.0 License, found in this directory.
This code takes inspiration from the RIAD paper (Zavrtanik et. al 2021)
"""

from typing import Any

import torch
import torch.nn.functional as F
from anomalib import LearningType
from anomalib.models.components import AnomalibModule
from torch import nn, optim


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """Downsample"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_ch), nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))

class Up(nn.Module):
    """Upsample"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))




class UNet(nn.Module):
    LOG_B_RANGE = (-6.0, 2.0)

    def __init__(self, nf=64):
        super().__init__()
        nc = 3
        self.uncertainty=True
        self.in_conv = DoubleConv(nc, nf)
        self.down1 = Down(nf, nf * 2)
        self.down2 = Down(nf * 2, nf * 4)
        self.down3 = Down(nf * 4, nf * 8)
        self.down4 = Down(nf * 8, nf * 8)

        self.up1 = Up(nf * 8, nf * 8, nf * 8)
        self.up2 = Up(nf * 8, nf * 4, nf * 4)
        self.up3 = Up(nf * 4, nf * 2, nf * 2)
        self.up4 = Up(nf * 2, nf, nf)
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nc, 3, 1, 1), nn.Tanh(),
        )
        if self.uncertainty:
            self.scale_conv = nn.Conv2d(nf, nc, 3, 1, 1)

    def forward(self, x, return_scale=False):
        s1 = self.in_conv(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        bottleneck = self.down4(s4)

        x = self.up1(bottleneck, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        out = self.out_conv(x)
        if not return_scale:
            return out
        return out, self.scale_conv(x).clamp(*self.LOG_B_RANGE)

def ssim_map(recon, target, window=11, sigma=1.5, data_range=2.0):
    channels = recon.shape[1]
    coords = torch.arange(window, device=recon.device, dtype=recon.dtype) - window // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).expand(channels, 1, window, window).contiguous()

    def blur(img):
        return F.conv2d(F.pad(img, [window // 2] * 4, mode="reflect"), kernel, groups=channels)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_recon = blur(recon) 
    mu_target = blur(target)
    var_recon = blur(recon * recon) - mu_recon ** 2
    var_target = blur(target * target) - mu_target ** 2
    cov = blur(recon * target) - mu_recon * mu_target
    return ((2 * mu_recon * mu_target + c1) * (2 * cov + c2)) / (
        (mu_recon ** 2 + mu_target ** 2 + c1) * (var_recon + var_target + c2))

def laplace_nll_loss(recon, log_b, target, masked, masked_weight=4.0, beta=0.5, ssim_weight=0.5):
    """Laplace negative log-likelihood

    beta (Seitzer et al. 2022, 'beta-NLL')
    """
    b = log_b.exp()
    l1 = (recon - target).abs()
    nll = l1 / b + log_b
    if beta > 0:
        nll = nll * b.detach() ** beta
    weight = 1.0 + (masked_weight - 1.0) * masked.expand_as(l1).float()
    loss = (nll * weight).mean()
    if ssim_weight > 0:
        loss = loss + ssim_weight * ((1.0 - ssim_map(recon, target)) * weight).mean()
    return loss

def random_block_mask(sample, mask_ratio, patch_size):
    x, _, h, w = sample.shape
    grid_y = -(-h // patch_size)
    grid_x = -(-w // patch_size)
    unmasked_ratio = 1.0 - mask_ratio
    grid = (torch.rand(x, 1, grid_y, grid_x, device=sample.device) < unmasked_ratio)
    keep_mask = F.interpolate(grid.float(), scale_factor=patch_size,
                                mode="nearest").bool()[..., :h, :w]
    masked = ~keep_mask
    return sample.masked_fill(masked, 0.0), masked

class UNetAnomalib(AnomalibModule):
    def __init__(self, nf=64, lr=1e-4, patch_sizes=None):
        super().__init__(pre_processor=False)
        self.save_hyperparameters()
        self.model = UNet(nf=nf)
        self.lr = lr
        self.patch_sizes = patch_sizes

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def to_neg_one_one(self, img: torch.Tensor) -> torch.Tensor:
        return img * 2.0 - 1.0 if img.min() >= 0.0 else img

    def training_step(self, batch, *args, **kwargs):
        target = self.to_neg_one_one(batch.image)
        patch_sizes = self.patch_sizes[torch.randint(len(self.patch_sizes), (1,)).item()]
        masked_input, mask = random_block_mask(target, mask_ratio=0.5, patch_size=patch_sizes)

        recon, log_b = self.model(masked_input, return_scale=True)
        loss = laplace_nll_loss(recon, log_b, target, mask).float()
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, *args, **kwargs):
        return

    def test_step(self, batch, *args, **kwargs):
        return