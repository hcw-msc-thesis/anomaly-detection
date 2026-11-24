import argparse
import os

os.environ["TORCH_LOGS"] = "-all"  # or set TORCH_CPP_LOG_LEVEL=ERROR

import warnings

import torch
from anomalib.callbacks import ModelCheckpoint
from anomalib.data import MVTec3D
from anomalib.engine import Engine
from torch.utils.data import DataLoader

from models.fpfh import (
    FPFHAnomalib,
)

warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="The anomalib.models.components.dinov2 package is deprecated", category=FutureWarning)

torch.set_float32_matmul_precision("high")

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Preprocessed dataset (background removed from the xyz tiffs)")
    p.add_argument("--category", required=True)
    p.add_argument("--image_size", type=int, default=224,
                   help="The grid the point cloud is resampled to (longest side)")
    p.add_argument("--patch", type=int, default=4,
                   help="Descriptors are averaged over patch x patch grid cells")
    p.add_argument("--normal_knn", type=int, default=30,
                   help="Nearest points used to estimate each surface normal")
    p.add_argument("--feature_knn", type=int, default=100,
                   help="Nearest points each FPFH descriptor is computed from")
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 3.0, 6.0],
                   help="Compute FPFH with the neighbour counts above times each of these and join them")
    p.add_argument("--min_coverage", type=float, default=0.5,
                   help="Skip patches with less than this fraction of object pixels")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="./checkpoints_fpfh",
                   help="Directory the bank goes in <out_dir>/<category>/")
    return p.parse_args(argv)


""" def stack_image_batch(items: list[torch.Tensor]) -> ImageBatch:
    return ImageBatch(image=torch.stack(items, dim=0))

class CacheDataset(Dataset):
    def __init__(self, dataset):
        self.images = [dataset[i].image.squeeze(0) for i in range(len(dataset))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx] """

class MVTec3DLoader(MVTec3D):
    def train_dataloader(self) -> DataLoader:
        loader = super().train_dataloader()
        dataset = loader.dataset
        return DataLoader(
            dataset=dataset,
            batch_size=loader.batch_size,
            shuffle=True,
            pin_memory=True,
            persistent_workers=False
        )

def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    data_dir = args.data_root
    data_cat = args.category

    dataset = MVTec3DLoader(
        root=data_dir,
        category=data_cat,
        train_batch_size=8,
        num_workers=4,
    )
    

    dataset.setup()

    datamodule = MVTec3D(
        root=data_dir,
        category=data_cat,
        train_batch_size=1,
        num_workers=4,
        val_split_mode="none",
    )
    datamodule.setup(stage="fit")


    model = FPFHAnomalib(
        patch=4,
        coreset_sampling_ratio=0.1,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{data_cat}/fpfh",
        filename="fpfh",
        save_last=False,
        enable_version_counter=False
    )

    engine = Engine(
        max_epochs=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )

    engine.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()