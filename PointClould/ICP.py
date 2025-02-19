#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: ICP.py
@time: 2025/2/19 12:48
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse

import numpy as np
import open3d as o3d
import cv2


NEW_DISTANCE = 50 # 50cm
ERROR_DISTANCE_RATIO = 0.1

def closest_point_cloud(source_points, target_points):
    """
    对源点云中的每个点，找到目标点云中的最近点，并返回这些匹配的点对。
    """
    target_tree = o3d.geometry.KDTreeFlann(target_points)
    matched_points = []

    for point in source_points:
        [k, idx, _] = target_tree.search_knn_vector_3d(point, 1)
        matched_points.append(target_points[idx[0]])

    return np.array(matched_points)


def compute_transform(source_points, target_points):
    """
    计算源点云到目标点云的刚性变换矩阵（旋转 + 平移）。
    """
    # 使用SVD计算最佳变换
    assert source_points.shape == target_points.shape
    centroid_src = np.mean(source_points, axis=0)
    centroid_tgt = np.mean(target_points, axis=0)

    # 去中心化
    source_centered = source_points - centroid_src
    target_centered = target_points - centroid_tgt

    # 计算协方差矩阵
    H = np.dot(source_centered.T, target_centered)

    # 奇异值分解
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)  # 旋转矩阵
    t = centroid_tgt - np.dot(R, centroid_src)  # 平移向量

    return R, t


def apply_transform(points, R, t):
    """
    将旋转矩阵和平移向量应用到点云，得到变换后的点云。
    """
    return np.dot(points, R.T) + t


def icp_registration(source_points, target_points, max_iterations=50, tolerance=1e-6):
    """
    ICP 配准算法的实现
    """
    prev_error = 0
    transformd_points = source_points.copy()

    for i in range(max_iterations):
        matched_points = closest_point_cloud(source_points, target_points)
        R, t = compute_transform(source_points, matched_points)

        source_points = apply_transform(source_points, R, t)
        mean_error = np.mean(np.linalg.norm(matched_points - source_points, axis=1))

        unmatched_points = source_points[np.linalg.norm(matched_points - source_points, axis=1) > NEW_DISTANCE]
        unmatched_points_transformed = apply_transform(unmatched_points, R, t)

        transformd_points = np.vstack([matched_points, unmatched_points_transformed])

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return transformd_points, R, t



def point_cloud_registration(points_list, colors_list):
    """
    处理当前帧的点云并与前一帧点云进行配准
    """
    for i, (points_3d, colors) in enumerate(zip(points_list, colors_list)):
        if i < 1:
            continue
        prev_points = points_list[i - 1]
        colors = colors_list[i - 1]
        points_3d_transformed, R, t = icp_registration(prev_points, points_3d)

        diff = np.linalg.norm(points_3d - prev_points, axis=1)
        diff /= prev_points
        valid_mask = diff < ERROR_DISTANCE_RATIO
        points_3d = points_3d[valid_mask]
        colors = colors[valid_mask]

        # 变换点云坐标
        # point_cloud_transformed = transform_point_cloud(points_3d, rotation, translation)

    return point_cloud_transformed, colors, points_3d  # 返回处理后的点云和前一帧的点云
