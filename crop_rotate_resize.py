"""
Once back is removed from the tiff files:
- Find rotation needed for each sample
- Rotate sample around its centre
- Find minimum bounding box needed for all training data, add 0.2 margin
- Apply rotations to the RGB and tiff files
- Store the transformed data in a new directory with a csv detailing the transformations done
"""

import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import tifffile
from anomalib.data.datasets.depth.mvtec_3d import make_mvtec_3d_dataset

from utils.crop_rotate import (
    apply_transformations,
    convert_matrix,
    find_rotation,
    image_dimensions,
    object_matrix,
)

ROTATION_METHOD = {
    "bagel": 0,
    "cable_gland": 2,
    "carrot": 1,
    "cookie": 0,
    "dowel": 2,
    "foam": 1,
    "peach": 0,
    "potato": 1,
    "rope": 1,
    "tire": 1
}

SPLITS = ["train", "validation", "test"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="e.g: ./mvtec_3d_anomaly_detection")
    p.add_argument("--category", required=True, help="e.g. dowel")
    p.add_argument("--out_dir", required=True, help="e.g. ./mvtec3d_test")
    p.add_argument("--normalise_brightness", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--image_size", required=True, default=224, type=int)
    return p.parse_args()

def write_files(sample, out_dir, crop_dims, resize, gt):
    write_tiffs(sample, out_dir, crop_dims, resize)
    write_images(sample, out_dir, crop_dims, resize)
    if gt is True:
        write_gt(sample, out_dir, crop_dims, resize)

def write_gt(sample, out_dir, crop_size, resize):
    image = sample["ground_truth"]
    crop_h, crop_w = crop_size
    resize_h, resize_w = resize
    
    resize_image = cv.resize(image, (resize_w, resize_h), interpolation=cv.INTER_AREA)
    h, w = resize_image.shape[:2]
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    cropped = resize_image[start_y : start_y + crop_h, start_x : start_x + crop_w]
    path = Path(out_dir) / sample["split"] / sample["label"] / "gt"
    image_name = Path(sample["file_name"]).stem + ".png"
    full_path = Path(path) / image_name
    full_path.parent.mkdir(parents=True, exist_ok=True)
    single_channel = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    cv.imwrite(full_path, single_channel)


def write_tiffs(sample, out_dir, crop_size, resize):
    mask = sample["transformed_mask"]
    crop_h, crop_w = crop_size
    resize_h, resize_w = resize

    resize_mask = cv.resize(mask, (resize_w, resize_h), interpolation=cv.INTER_NEAREST_EXACT)
    h, w = resize_mask.shape[:2]
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    cropped = resize_mask[start_y : start_y + crop_h, start_x : start_x + crop_w]
    path = Path(out_dir) / sample["split"] / sample["label"] / "xyz"
    mask_name = Path(sample["file_name"]).stem + ".tiff"
    full_path = Path(path) / mask_name
    full_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(full_path, cropped)

def write_images(sample, out_dir, crop_size, resize):
    image = sample["transformed_image"]
    crop_h, crop_w = crop_size
    resize_h, resize_w = resize
    
    resize_image = cv.resize(image, (resize_w, resize_h), interpolation=cv.INTER_AREA)
    h, w = resize_image.shape[:2]
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    cropped = resize_image[start_y : start_y + crop_h, start_x : start_x + crop_w]
    path = Path(out_dir) / sample["split"] / sample["label"] / "rgb"
    image_name = Path(sample["file_name"]).stem + ".png"
    full_path = Path(path) / image_name
    full_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(full_path, cropped)

def get_image_brightness(img):
    lab_colour = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel = lab_colour[:, :, 0]
    brightness = float(np.mean(l_channel))
    return brightness

def normalise_brightness(img, target_brightness):
    lab_colour = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    current_brightness = np.mean(lab_colour[:, :, 0])
    scale = target_brightness / current_brightness
    lab_colour[:, :, 0] = np.clip(lab_colour[:, :, 0].astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return cv.cvtColor(lab_colour, cv.COLOR_LAB2BGR)
    

def main():
    args = parse_args()
    cat_dir = os.path.join(args.data_root, args.category)
    out_cat = os.path.join(args.out_dir, args.category)
    rotation = ROTATION_METHOD[args.category]

    training_samples = make_mvtec_3d_dataset(cat_dir, split="train")
    testing_samples = make_mvtec_3d_dataset(cat_dir, split="test")
    validation_samples = make_mvtec_3d_dataset(cat_dir, split="validation")

    
    (original_height, original_width) = image_dimensions(training_samples.iloc[0]["depth_path"])
    original_dimensions = (original_width, original_height)
    training_samples["rotation"] = training_samples["depth_path"].apply(find_rotation, args=(rotation,))
    #Rotate samples first
    transforms = training_samples.apply(
        lambda row: object_matrix(row["depth_path"], row["rotation"], original_dimensions), axis=1)
    matrix, box = zip(*transforms)
    training_samples["matrix"] = matrix
    image_data = training_samples[["image_path", "depth_path", "matrix"]].copy()
    results = image_data.apply(lambda row: apply_transformations(
        row["image_path"],
        row["depth_path"],
        row["matrix"],
        original_dimensions,
    ),
    axis=1,)
    images, masks, _ = zip(*results)
    width, height = zip(*box)
    crop_width = int(max(width) * 1.2)
    crop_width = min(crop_width, original_width)
    crop_height = int(max(height) * 1.2)
    crop_height = min(crop_height, original_height)
    print(crop_height)
    print(crop_width)
    multiple = 16
    image_size = args.image_size
    if crop_height >= crop_width:
        scale = crop_height / image_size
        temp_width = crop_width / scale
        new_height = image_size
        new_width = multiple * round(temp_width / multiple)
    else:
        scale = crop_width / image_size
        temp_height = crop_height / scale
        new_width = image_size
        new_height = multiple * round(temp_height / multiple)
    scaled_h = round(original_height / scale)
    scaled_w = round(original_width / scale)

    resize = (scaled_h, scaled_w)
    crop = (new_height, new_width)
    print(resize)
    print(crop)
    training_samples["transformed_image"] = images
    training_samples["transformed_mask"] = masks
    training_samples["brightness"] = training_samples["transformed_image"].apply(get_image_brightness)
    median_brightness = training_samples["brightness"].median()
    if args.normalise_brightness:
        training_samples["transformed_image"] = training_samples["transformed_image"].apply(normalise_brightness, args=(median_brightness,))
    training_samples.apply(write_files, axis=1, args=(out_cat, crop, resize, False))
    training_samples = training_samples.drop(columns=["path", "image_path", "depth_path", "mask_path", "transformed_image", "transformed_mask", "brightness"])
    """ csv_path = Path(out_cat) / "rotations.csv"
    training_samples.to_csv(csv_path, index=False) """


    
    #testing_samples = datamodule.test_data.samples
    testing_samples["rotation"] = testing_samples["depth_path"].apply(find_rotation, args=(rotation,))
    test_transforms = testing_samples.apply(
        lambda row: object_matrix(row["depth_path"], row["rotation"], original_dimensions), axis=1)
    test_matrix, _ = zip(*test_transforms)
    testing_samples["matrix"] = test_matrix
    testing_image_data = testing_samples[["image_path", "depth_path", "mask_path", "matrix"]].copy()
    test_results = testing_image_data.apply(lambda row: apply_transformations(
        row["image_path"],
        row["depth_path"],
        row["matrix"],
        original_dimensions,
        row["mask_path"],
    ),
    axis=1,)
    test_images, test_masks, test_gt = zip(*test_results)
    testing_samples["transformed_image"] = test_images
    testing_samples["transformed_mask"] = test_masks
    testing_samples["ground_truth"] = test_gt
    if args.normalise_brightness:
        testing_samples["transformed_image"] = testing_samples["transformed_image"].apply(normalise_brightness, args=(median_brightness,))
    testing_samples.apply(write_files, axis=1, args=(out_cat, crop, resize, True))
    testing_samples[["m00", "m01", "m02", "m10", "m11", "m12"]] = testing_samples["matrix"].apply(convert_matrix)
    testing_samples["original_dimensions"] = original_height
    testing_samples = testing_samples.drop(columns=["path", "image_path", "depth_path", "mask_path", "transformed_image", "transformed_mask", "matrix", "ground_truth"])
    csv_path = Path(out_cat) / "rotations.csv"
    testing_samples["scale_h"] = original_height / scaled_h
    testing_samples["scale_w"] = original_width / scaled_w
    testing_samples.to_csv(csv_path, index=False)

    validation_samples["rotation"] = validation_samples["depth_path"].apply(find_rotation, args=(rotation,))
    validation_transforms = validation_samples.apply(
        lambda row: object_matrix(row["depth_path"], row["rotation"], original_dimensions), axis=1)
    validation_matrix, _ = zip(*validation_transforms)
    validation_samples["matrix"] = validation_matrix
    validation_image_data = validation_samples[["image_path", "depth_path", "matrix"]].copy()
    validation_results = validation_image_data.apply(lambda row: apply_transformations(
        row["image_path"],
        row["depth_path"],
        row["matrix"],
        original_dimensions,
    ),
    axis=1,)
    validation_images, validation_masks, _ = zip(*validation_results)
    validation_samples["transformed_image"] = validation_images
    validation_samples["transformed_mask"] = validation_masks
    if args.normalise_brightness:
        validation_samples["transformed_image"] = validation_samples["transformed_image"].apply(normalise_brightness, args=(median_brightness,))
    validation_samples.apply(write_files, axis=1, args=(out_cat, crop, resize, False))

if __name__ == "__main__":
    main()