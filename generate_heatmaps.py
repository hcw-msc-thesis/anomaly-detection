"""
Generate heatmaps

- Evaluate UNet, get heatmaps
    - Call on unet.py?
- Evaluate FPFH, get heatmaps
    - Call on evaluate_
- Normalise and combine them
- Undo their rotation
- Save the heatmaps
- Evaluation scores by MVTec

"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from anomalib.data import MVTec3D, MVTec3DDataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from models.fpfh import FPFHAnomalib
from models.unet import UNetAnomalib, random_block_mask
from utils.crop_rotate import undo_transformations
from utils.unet_errors import unet_error

CATEGORIES = [
    "bagel",
    "cable_gland",
    "carrot",
    "cookie",
    "dowel",
    "foam",
    "peach",
    "potato",
    "rope",
    "tire"
]

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Output of preprocess_dataset.py")
    p.add_argument("--category", required=False)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--unet_patch_sizes", type=int, nargs="+", default=None,
                   help="Multi-scale masking: each batch uses one of these block sizes at random and evaluation averages their error maps")
    p.add_argument("--unet_errors", default=["l1", "ssim", "chroma", "unc_l1", "unc_ssim", "unc_chroma"], nargs="+",
                       choices=["l1", "ssim", "chroma", "unc_l1", "unc_ssim", "unc_chroma"],)
    p.add_argument("--unet_unc_floor_pct", type=float, default=10,
                       help="unc_* errors: never divide by less than this percentile")
    p.add_argument("--unet_n_masks", type=int, default=4,
                       help="U-Net: average this many different block layouts per block size "
                            "(see detect_unet.error_channel_maps); slower")
    p.add_argument("--out_dir", default="./heatmaps")
    return p.parse_args(argv)


def object_mask(batch):
    mask = (batch.depth_map[:, 2] != 0)
    return mask

@torch.no_grad()
def unet_anomaly_map(
    unet,
    image_batch,
    unet_errors,
    uncertainty_floor,
    n_masks,
    patch_sizes,
):
    target = unet.to_neg_one_one(image_batch)
    error_totals = {error: 0 for error in unet_errors}
    b_total = 0
    for patch_size in patch_sizes:

        for _ in range(n_masks):

            _, mask_1 = random_block_mask(target, 0.5, patch_size)
            mask_2 = ~mask_1

            recon_1, log_b_1 = unet.model(target.masked_fill(mask_1, 0.0), return_scale=True)
            recon_2, log_b_2 = unet.model(target.masked_fill(mask_2, 0.0), return_scale=True)

            for error in unet_errors:
                error_1 = unet_error(target, recon_1, log_b_1, uncertainty_floor, error)
                error_2 = unet_error(target, recon_2, log_b_2, uncertainty_floor, error)

                combined_error = torch.where(mask_1, error_1, error_2)
                error_totals[error] += combined_error

            b = torch.where(mask_1, log_b_1.exp(), log_b_2.exp())
            b_total += b

    divider = len(patch_sizes) * n_masks
    averaged_b = (b_total / divider).mean(dim=1)
    averaged_errors = {error: (total / divider).mean(dim=1) for error, total in error_totals.items()}
    return averaged_errors, averaged_b


@torch.no_grad()
def fpfh_anomaly_map(fpfh, depth_batch):
    fpfh.model.memory_bank = fpfh.model.memory_bank.cpu()
    preds = fpfh.model(depth_batch.detach().cpu())
    return preds.anomaly_map.squeeze(1)


@torch.no_grad()
def model_stats(unet, fpfh, validation_loader, device, unet_errors, uncertainty_pct, n_masks, patch_sizes):
    unet_scores = {error: [] for error in unet_errors}
    fpfh_scores = []
    b_scores = []

    for batch in tqdm(validation_loader, desc="Calculating uncertainty floor"):
        _, b = unet_anomaly_map(unet, batch.image.to(device), unet_errors=[], uncertainty_floor=0, n_masks=n_masks, patch_sizes=patch_sizes)
        b_scores.append(b.cpu()[object_mask(batch)])

    all_b = torch.cat(b_scores, dim=0)
    b_floor = torch.quantile(all_b, uncertainty_pct / 100).item()
    
        
    for batch in tqdm(validation_loader, desc="Calibrating normal stats"):
        mask = object_mask(batch)
        dict_unet, _ = unet_anomaly_map(unet, batch.image.to(device), unet_errors, uncertainty_floor=b_floor, n_masks=n_masks, patch_sizes=patch_sizes,)
        map_fpfh = fpfh_anomaly_map(fpfh, batch.depth_map).cpu()

        unet_blurred_dict = {error: gaussian_blur(tensor.cpu(), kernel_size=[13, 13], sigma=[2.0, 2.0]) for error, tensor in dict_unet.items()}
        fpfh_blurred = gaussian_blur(map_fpfh.cpu(), kernel_size=[25, 25], sigma=[4.0, 4.0])

        for error, unet_blurred in unet_blurred_dict.items():
            unet_scores[error].append(unet_blurred[mask])
        fpfh_scores.append(fpfh_blurred[mask])

    all_unet = {error: torch.cat(values) for error, values in unet_scores.items()}
    all_fpfh = torch.cat(fpfh_scores, dim=0)
    unet_mean_std = {
        error: {
            "mean": tensor.mean().item(), 
            "std": tensor.std().clamp(min=1e-6).item()
        } 
        for error, tensor in all_unet.items()}
    
    standardised_unet_dict = {
        error: (tensor - unet_mean_std[error]["mean"]) / unet_mean_std[error]["std"]
        for error, tensor in all_unet.items()}

    stacked_unet = torch.stack(list(standardised_unet_dict.values()), dim=0)
    combined_unet = stacked_unet.sum(dim=0)
    
    return {
        "unet_errors": unet_mean_std,
        "unet_combined": {
            "mean": combined_unet.mean().item(),
            "std": combined_unet.std().clamp(min=1e-6).item(),
        },
        "mean_fpfh": all_fpfh.mean().item(),
        "std_fpfh": all_fpfh.std().clamp(min=1e-6).item(),
        "uncertainty_floor": b_floor
    }


@torch.no_grad()
def export_heatmaps(
    category,
    dataset_root = "./mvtec3d_preprocessed",
    output_dir = "./anomaly_heatmaps",
    patch_sizes = None,
    unet_errors = None,
    n_masks = 4,
    uncertainty_pct = 10,
    weight_unet = 1,
    weight_fpfh = 1,
    seed = 0
):
    if patch_sizes is None:
        patch_sizes = [8, 16, 32]
    if unet_errors is None:
        unet_errors = ["ssim", "l1", "chroma", "unc_ssim", "unc_l1", "unc_chroma"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    path = f"./checkpoints/{category}"
    unet_path = f"{path}/unet/unet.ckpt"
    fpfh_path = f"{path}/fpfh/fpfh.ckpt"
    unet = UNetAnomalib.load_from_checkpoint(unet_path, strict=False).to(device).eval()


    checkpoint_data = torch.load(fpfh_path, map_location="cpu", weights_only=False)

    fpfh = FPFHAnomalib(**checkpoint_data.get("hyper_parameters", {}))
    fpfh.model.memory_bank = checkpoint_data["state_dict"]["model.memory_bank"]
    fpfh.model.block_spreads = checkpoint_data["state_dict"]["model.block_spreads"]
    fpfh.eval()

    datamodule = MVTec3D(
        root=dataset_root,
        category=category,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        seed=seed
    )

    validation = MVTec3DDataset(
        root=dataset_root,
        category=category,
        split="validation"
    )

    dataloader = DataLoader(dataset=validation, shuffle=False, collate_fn=validation.collate_fn, batch_size=8)
    
    datamodule.setup()

    stats = model_stats(unet, fpfh, dataloader, device, unet_errors, uncertainty_pct, n_masks, patch_sizes)
    uncertainty_floor = stats["uncertainty_floor"]

    rotations = pd.read_csv(Path(dataset_root) / category / "rotations.csv")
    cols = ["m00", "m01", "m02", "m10", "m11", "m12"]
    matrices = rotations[cols].to_numpy().reshape(-1, 2, 3)
    rotations["matrix"] = list(matrices)

    file_path = ["split", "label", "type", "file_name"]
    rotations["path"] = rotations[file_path].astype(str).agg("/".join, axis=1)

    for batch in tqdm(datamodule.test_dataloader(), desc=f"Exporting {category} heatmaps"):
        
        unet_heatmap_dict, _ = unet_anomaly_map(unet, batch.image.to(device), unet_errors=unet_errors, uncertainty_floor=uncertainty_floor, n_masks=n_masks, patch_sizes=patch_sizes)
        fpfh_heatmap = fpfh_anomaly_map(fpfh, batch.depth_map).cpu()
            
        unet_blurred_dict = {error: gaussian_blur(tensor.cpu(), kernel_size=[13, 13], sigma=[2.0, 2.0]) for error, tensor in unet_heatmap_dict.items()}
        fpfh_blurred = gaussian_blur(fpfh_heatmap.cpu(), kernel_size=[25, 25], sigma=[4.0, 4.0])


        norm_unet_dict = {
            error: (tensor - stats["unet_errors"][error]["mean"]) / stats["unet_errors"][error]["std"]
                for error, tensor in unet_blurred_dict.items()}
        
        norm_fpfh = (fpfh_blurred - stats["mean_fpfh"]) / stats["std_fpfh"]

        stacked_unet = torch.stack(list(norm_unet_dict.values()), dim=0)
        combined_unet = stacked_unet.sum(dim=0)
        norm_unet = (combined_unet - stats["unet_combined"]["mean"]) / stats["unet_combined"]["std"]
        
        


        combined_maps = (weight_unet * norm_unet + weight_fpfh * norm_fpfh)
        maps_output = combined_maps.numpy()

        for img_path, heatmap in zip(batch.image_path, maps_output):
            path_end = img_path.split(category + "/")[-1]
            row = rotations[rotations["path"] == path_end].iloc[0]
            transformed = undo_transformations(row["original_dimensions"], row["matrix"], heatmap, (0,0), (row["scale_h"], row["scale_w"]))

            parts = os.path.normpath(img_path).split(os.sep)
            defect_type = parts[-3]
            file_stem = os.path.splitext(parts[-1])[0]

            save_folder = os.path.join(output_dir, category, "test", defect_type)
            os.makedirs(save_folder, exist_ok=True)
            heatmap_path = os.path.join(save_folder, f"{file_stem}.tiff")
            tifffile.imwrite(heatmap_path, transformed.astype(np.float32))


def main(argv=None):
    args = parse_args(argv)
    if args.category is None:
        for category in CATEGORIES:
            export_heatmaps(category=category, 
                            dataset_root=args.data_root, 
                            output_dir=args.out_dir,
                            patch_sizes=args.unet_patch_sizes,
                            unet_errors=args.unet_errors,
                            n_masks=args.unet_n_masks, 
                            uncertainty_pct=args.unet_unc_floor_pct,
                            )
    else:
        export_heatmaps(category=args.category, 
                        dataset_root=args.data_root, 
                        output_dir=args.out_dir,
                        patch_sizes=args.unet_patch_sizes,
                        unet_errors=args.unet_errors,
                        n_masks=args.unet_n_masks, 
                        uncertainty_pct=args.unet_unc_floor_pct,
                        )


if __name__ == "__main__":
    main()




#TODO

# Add the error types
# Be able to select the error types
# Make sure this actually works properly
# 