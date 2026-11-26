import argparse
import os

os.environ["TORCH_LOGS"] = "-all"  # or set TORCH_CPP_LOG_LEVEL=ERROR

import warnings

import torch
from anomalib.callbacks import ModelCheckpoint
from anomalib.data import MVTec3D
from anomalib.data.dataclasses import ImageBatch
from anomalib.engine import Engine
from torch.utils.data import DataLoader, Dataset

from unet.unet import (
    UNetAnomalib,
)

warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="The anomalib.models.components.dinov2 package is deprecated", category=FutureWarning)

torch.set_float32_matmul_precision("high")

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Output of preprocess_dataset.py")
    p.add_argument("--category", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--nf", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--patch_sizes", type=int, nargs="+", default=None,
                   help="Multi-scale masking: each batch uses one of these block sizes at random and evaluation averages their error maps")
    p.add_argument("--beta", type=float, default=0.5,
                   help="--uncertainty: beta-NLL weight (0 = plain NLL, 1 = plain L1)")
    p.add_argument("--masked_weight", type=float, default=4.0)
    p.add_argument("--ssim_weight", type=float, default=0.5,
                   help="add this x (1 - SSIM) to the L1 loss")
    p.add_argument("--out_dir", default="./checkpoints",
                   help="Checkpoint output directory")
    p.add_argument("--sample_every", type=int, default=5,
                   help="Display per epoch sample")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def stack_image_batch(items: list[torch.Tensor]) -> ImageBatch:
    return ImageBatch(image=torch.stack(items, dim=0))

class CacheDataset(Dataset):
    def __init__(self, dataset):
        self.images = [dataset[i].image.squeeze(0) for i in range(len(dataset))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]

class MVTec3DLoader(MVTec3D):
    def train_dataloader(self) -> DataLoader:
        loader = super().train_dataloader()
        dataset = CacheDataset(loader.dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=loader.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=stack_image_batch,
            pin_memory=True,
            persistent_workers=False,
            drop_last=True
        )

def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu (torch.cuda.is_available() returned False -- "
                  "check your PyTorch/CUDA install if you expected a GPU here)")
    data_dir = args.data_root
    data_cat = args.category
    batch_size = args.batch_size
    out_dir = os.path.join(args.out_dir, args.category, "unet")
    os.makedirs(out_dir, exist_ok=True)

    dataset = MVTec3DLoader(
        root=data_dir,
        category=data_cat,
        train_batch_size=batch_size,
        seed=args.seed,
        num_workers=args.workers,
    )
    

    dataset.setup()


    model = UNetAnomalib(
        nf=args.nf,
        lr=args.lr,
        patch_sizes=args.patch_sizes
    )

    callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{args.category}/unet",
        filename="unet",
        save_last=False,
        every_n_epochs=20,
        save_weights_only=True,
        enable_version_counter=False
    )

    engine = Engine(
        callbacks=[callback],
        max_epochs=args.epochs,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        enable_checkpointing=True,
    )

    engine.fit(model=model, datamodule=dataset)
    #engine.test(model=model, datamodule=dataset)

if __name__ == "__main__":
    main()