#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: eval.py
@time: 2025/1/23 15:16
@desc: 
'''
import sys, os
from os import pread

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse

import cv2
from tqdm import tqdm
import numpy as np
from utils.file import match_images
from PointClould.ICP2D import ScaleShiftAnalyzer
from utils.file import MkdirSimple

MIN = 0.0001
MAX = 65535.0 - 1
MAX_DISTANCE4Save = 700.0

def GetArgs():
    parser = argparse.ArgumentParser(description="Depth Estimation Evaluation Tool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gt", type=str, required=True, help="Directory containing ground truth depth maps")
    parser.add_argument("--pred", type=str, required=True, help="Directory containing predicted depth maps")
    parser.add_argument("--rgb", type=str, help="Directory containing rgb image")
    parser.add_argument("--show", action="store_true", help="Show result")
    parser.add_argument("--save", type=str, help="Directory to save aligned depth maps")
    parser.add_argument("--max_dist", type=int, default=500, help="max distance for evaluation (cm)")

    args = parser.parse_args()
    return args

def compute_errors(gt, pred, mask=None):
    gt = gt + 1e-12
    pred = pred + 1e-12
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]

    abs_diff = np.mean(np.abs(gt - pred))
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    a0 = np.mean((np.maximum(gt / pred, pred / gt) < 1.1).astype(np.float32))
    a1 = np.mean((np.maximum(gt / pred, pred / gt) < 1.25).astype(np.float32))
    a2 = np.mean((np.maximum(gt / pred, pred / gt) < 1.25 ** 2).astype(np.float32))
    a3 = np.mean((np.maximum(gt / pred, pred / gt) < 1.25 ** 3).astype(np.float32))
    return [abs_diff, abs_rel, a1, a2, a3, a0]


def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

def normalize(image):
    image = image.astype("float32")
    image = (image - image.min()) / (image.max() - image.min()) * 255.0
    image = image.astype("uint8")
    return image

def visualize_image(rgb, gt, pred, min_invalid, name="result"):
    mask = (gt > MIN) & (gt < min_invalid)
    gt[gt >= min_invalid] = min_invalid
    pred[pred >= min_invalid] = min_invalid

    # 可视化原始图像、处理后的 mask 图像和差异图像
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt


    cmap = 'magma'

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(name)

    axes[0, 0].imshow(rgb, cmap=None)
    axes[0, 0].set_title('original image')
    diff = np.abs(gt.astype(np.float16) - pred.astype(np.float16)).astype(np.uint64)
    diff[~mask] = 0
    diff_rel = diff / gt
    axes[1, 0].imshow(diff, cmap=cmap)
    axes[1, 0].set_title(f'Difference {np.mean(diff[mask]):.2f} / DiffRel {np.mean(diff_rel[mask]):.2f}')


    axes[0, 1].imshow(gt, cmap=cmap)
    axes[0, 1].set_title('Ground Truth')
    axes[1, 1].imshow(pred, cmap=cmap)
    axes[1, 1].set_title('Predicted Aligned')

    axes[0, 2].imshow(np.where(mask, gt, np.nan), cmap=cmap)
    axes[0, 2].set_title('Ground Truth - Masked')
    axes[1, 2].imshow(np.where(mask, pred, np.nan), cmap=cmap)
    axes[1, 2].set_title('Predicted Aligned - Masked')

    plt.show()

def evaluate_depth_maps(ground_truth_dir, predicted_dir, rgb_dir, show, save_dir, MAX_DISTANCE4Eval):
    if not rgb_dir:
        gt_images, pred_images = match_images([ground_truth_dir, predicted_dir])
        rgb_images = [None] * len(gt_images)
    else:
        gt_images, pred_images, rgb_images = match_images([ground_truth_dir, predicted_dir, rgb_dir])

    abs_diff_total = 0
    abs_rel_total = 0
    a0_total = 0
    a1_total = 0
    a2_total = 0
    a3_total = 0
    count = 0
    count_invalid = 0
    recall = 0

    icp2d = ScaleShiftAnalyzer()

    for gt_path, pred_path, rgb_path in tqdm(zip(gt_images, pred_images, rgb_images), total=len(gt_images), desc="Processing images"):
        gt = load_image(gt_path)
        pred = load_image(pred_path)

        # to cm
        gt_cm = (gt / 255.0  * 100 * 0.25).astype(np.uint16) # to cm
        # gt_cm = (gt / 65535.0 * 20  * 100).astype(np.uint16) # to cm
        mask_gt = (gt_cm > MIN) & (gt_cm < MAX_DISTANCE4Eval) & (gt > MIN) & (gt < 65535)
        mask_pred = (pred > MIN) & (pred < 65535)
        mask = mask_gt & mask_pred

        sum_gt = np.sum(mask_gt)
        if sum_gt == 0 or np.sum(mask) == 0:
            count_invalid += 1
            continue

        pred_missing_ratio = np.sum(mask) / np.sum(mask_gt)
        recall += pred_missing_ratio

        # todo: hao 2025-02-04 15:45 - why scale
        gt = gt * 10.0
        gt[gt > 65535] = 65535
        gt = gt.astype(np.uint16)

        scale, shift = icp2d.compute_scale_and_shift(pred, gt, mask)
        pred_aligned = icp2d.align(pred, scale, shift)

        errs = compute_errors(gt, pred_aligned, mask)
        abs_diff_total += errs[0]
        abs_rel_total += errs[1]
        a1_total += errs[2]
        a2_total += errs[3]
        a3_total += errs[4]
        a0_total += errs[5]
        count += 1

        if show:
            print(errs)
            name = '/'.join(gt_path.split('/')[-3:])
            rgb = load_image(rgb_path) if rgb_path is not None else np.zeros_like(gt)
            min_invalide = gt[gt_cm >= MAX_DISTANCE4Eval].min() if (gt_cm >= MAX_DISTANCE4Eval).any() else gt.max()
            visualize_image(rgb, gt, pred_aligned, min_invalide, name)

        if save_dir:
            # todo: hao 2025-01-24 00:12 - how to process the max value/max invalid region
            # pred_aligned[mask_gt] = gt[mask_gt]
            save_path = os.path.join(save_dir, pred_path[len(predicted_dir.rstrip('/'))+1:])
            MkdirSimple(save_path)
            mask_max = pred == 65535
            mask_invalid = pred <= MIN
            pred = pred.astype(np.float32)
            pred_aligned = icp2d.align(pred, scale, shift)
            mask_invalid = mask_invalid | (pred_aligned <= MIN)
            max_value = pred_aligned[mask_max].max() if mask_max.any() else 65535
            max_value = min(max_value, 65535)
            pred_aligned[pred_aligned >= max_value] = max_value
            pred_aligned[mask_invalid] = 0
            pred_aligned = pred_aligned.astype("uint16")
            cv2.imwrite(save_path, pred_aligned)

    abs_diff_avg = abs_diff_total / count
    abs_rel_avg = abs_rel_total / count
    a1_avg = a1_total / count
    a2_avg = a2_total / count
    a3_avg = a3_total / count
    a0_avg = a0_total / count
    recall = recall / count

    errs = {'abs_diff': abs_diff_avg, 'abs_rel': abs_rel_avg,
            'a0': a0_avg, 'a1': a1_avg, 'a2': a2_avg, 'a3': a3_avg,
            'recall': recall}

    print("Errors:")
    for key, value in errs.items():
        print(f"  {key}: {value:.6f}")

    print(f"Invalid item count: {count_invalid}")
    icp2d.plot_scale_and_shift()

def main():
    args = GetArgs()
    evaluate_depth_maps(args.gt, args.pred, args.rgb, args.show, args.save, args.max_dist)


if __name__ == '__main__':
    main()
