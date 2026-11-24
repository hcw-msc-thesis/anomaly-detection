"""
Draw the object's outline, smallest rotated box and fitted ellipse on an RGB
image, using the object mask from its BTF-preprocessed XYZ tiff (every pixel
with a point left is object). Adapted from OpenCV's "Creating Bounding rotated
boxes and ellipses for contours" tutorial.
"""


import cv2 as cv
import numpy as np
import pandas as pd
import tifffile


def object_mask(xyz_path, close_size=15):
    xyz = tifffile.imread(xyz_path)
    mask = (xyz[..., 2] > 0).astype(np.uint8) * 255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_size, close_size))
    return cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

def img_file(rgb_path):
    rgb = cv.imread(rgb_path)
    return rgb


def draw_object_box(src, mask, color=(0, 255, 0)):
    """Copy of src with the largest outline in mask, its rotated box and ellipse drawn on."""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SystemExit("No mask found")
    largest = max(contours, key=cv.contourArea)

    drawing = src.copy()
    cv.drawContours(drawing, [largest], 0, color, 1)
    box = np.intp(cv.boxPoints(cv.minAreaRect(largest)))
    cv.drawContours(drawing, [box], 0, color, 2)
    if len(largest) > 5:
        cv.ellipse(drawing, cv.fitEllipse(largest), (0, 0, 255), 2)

def get_largest(mask_path):
    mask = object_mask(mask_path)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv.contourArea)
    return largest

def object_centre(mask_path):
    largest = get_largest(mask_path)
    (cx, cy), (_, _), _ = cv.minAreaRect(largest)
    return (cx, cy)

def object_matrix (mask_path, rotation, img_dims):
    centre = object_centre(mask_path)
    img_w, img_h = img_dims
    M = cv.getRotationMatrix2D(centre, rotation, 1.0)
    largest = get_largest(mask_path)
    x, y, w, h = cv.boundingRect(cv.transform(largest.astype(np.float32), M))
    M[0, 2] += img_w / 2 - (x + w / 2)
    M[1, 2] += img_h / 2 - (y + h / 2)
    return M, (w, h)



def find_rotation(mask_path, rotation_rule):
    mask = object_mask(mask_path)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv.contourArea)
    (cx, cy), (_, _), angle = cv.minAreaRect(largest)
    match rotation_rule:
        case 0:
            return 0
        case 1:
            rotation = ((angle+45) % 90) - 45
            return (rotation)
        case 2:
            (ex, ey), (major_axis, minor_axis), ellipse_angle = cv.fitEllipse(largest)
            long_angle = ellipse_angle if major_axis >= minor_axis else ellipse_angle - 90
            
            long_angle = (long_angle + 90) % 180 - 90       
            t = np.radians(long_angle)
            dot_prod = (ex - cx) * np.cos(t) + (ey - cy) * np.sin(t)
            if dot_prod < 0:
                long_angle += 180
            rotation_angle = (long_angle + 180) % 360

            return (rotation_angle)

def find_box_dimensions(mask_path):
    mask = object_mask(mask_path)
    points = cv.findNonZero(mask)
    rect = cv.boundingRect(points)
    return rect

def image_dimensions(mask_path):
    mask = object_mask(mask_path)
    shape = mask.shape
    return shape

def image_translation(rotation_point, img_dimensions):
    h, w = img_dimensions[:2]
    diff_x = w / 2 - rotation_point[0]
    diff_y = h / 2 - rotation_point[1]
    return (diff_x, diff_y)

def convert_matrix(matrix):
    raveled = matrix.ravel()
    return pd.Series(raveled)

def apply_transformations(img_path, mask_path, M, img_dims, gt_path=None):
    flag_list=cv.INTER_NEAREST
    image = cv.warpAffine(img_file(img_path), M, img_dims)
    mask = cv.warpAffine(tifffile.imread(mask_path), M, img_dims, flags=flag_list)
    if pd.notna(gt_path):
        gt = cv.warpAffine(img_file(gt_path), M, img_dims)
    else:
        gt = None
    return (image, mask, gt)

def transform_img(img_path, rotation, matrix, img_dims, img_centre, inverse=False):
    img = img_file(img_path)
    translated_image = cv.warpAffine(img, matrix, img_dims)
    rotation_matrix = cv.getRotationMatrix2D(img_centre, rotation, 1.0)
    rotated_image = cv.warpAffine(translated_image, rotation_matrix, img_dims)

    return rotated_image

def transform_tiff(mask_path, rotation, matrix, img_dims, img_centre):
    mask = tifffile.imread(mask_path)
    flag_list=cv.INTER_NEAREST
    translated_mask = cv.warpAffine(mask, matrix, img_dims, flags=flag_list)
    rotation_matrix = cv.getRotationMatrix2D(img_centre, rotation, 1.0)
    rotated_mask = cv.warpAffine(translated_mask, rotation_matrix, img_dims, flags=flag_list)

    return rotated_mask

def undo_transformations(original_dim, M, mask, gt_dims, scale):
    #mask = tifffile.imread(mask_path)
    scale_h, scale_w = scale
    orig_w = original_dim
    orig_h = original_dim
    mask_h, mask_w = mask.shape[:2]
    pad_w = round(mask_w * scale_w)
    pad_h = round(mask_h * scale_h)
    resize_mask = cv.resize(mask, (pad_w, pad_h), interpolation=cv.INTER_LINEAR)
    mask_h, mask_w = resize_mask.shape[:2]
    gt_w, gt_h = gt_dims
    start_y = (orig_h - mask_h) // 2
    start_x = (orig_w - mask_w) // 2
    pad_tiff = np.full((orig_h, orig_w, *resize_mask.shape[2:]), resize_mask.min() - 1, dtype=resize_mask.dtype)
    pad_tiff[start_y : start_y + mask_h, start_x : start_x + mask_w] = resize_mask
    M_new = np.asarray(M, dtype=np.float64).reshape(2, 3)
    #M_inv = cv.invertAffineTransform(np.asarray(M, dtype=np.float64).reshape(2, 3))
    undo = cv.warpAffine(
        pad_tiff,
        M_new,
        (orig_w, orig_h),
        flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    return undo
    uncrop_start_y = (orig_h - gt_h) // 2
    uncrop_start_x = (orig_w - gt_w) // 2

    output = undo[uncrop_start_y : uncrop_start_y + gt_h, uncrop_start_x : uncrop_start_x + gt_w]

    return output
