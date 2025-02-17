#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: scale_depth.py
@time: 2025/2/17 09:31
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse

import numpy as np
from scipy.optimize import least_squares
from utils.file import ReadImageList, MkdirSimple, match_images
import cv2
from utils.dataset import ConfigLoader
from utils.utils import timeit
from PointClould.ICP2D import ScaleShiftAnalyzer

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--depth", type=str, help="depth image dir or file path or list of depth image files")
    parser.add_argument("--left", type=str, help="left image dir or file path or list of depth image files")
    parser.add_argument("--right", type=str, help="right image dir or file path or list of depth image files")
    parser.add_argument("--output", type=str, help="scaled depth image dir or file path")

    args = parser.parse_args()
    return args


def project_point(K, X_cam):
    """
    将相机坐标系下的 3D 点 [X, Y, Z] 投影到图像平面
    假设已无畸变或已做矫正。
    输入:
      K: 3x3 内参矩阵
      X_cam: shape=(3,) or (3,1) 的 3D 坐标
    输出:
      (u, v): 投影像素坐标
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X, Y, Z = X_cam[0], X_cam[1], X_cam[2]
    # 透视投影
    u = (fx * X / Z) + cx
    v = (fy * Y / Z) + cy
    return (u, v)

def transform_point(T, X):
    """
    将 3D 点 X (shape=(3,)) 用 4x4 齐次矩阵 T 进行坐标变换
    返回变换后的 3D 坐标
    """
    X_h = np.array([X[0], X[1], X[2], 1.0])
    X_h_new = T.dot(X_h)
    return X_h_new[:3] / X_h_new[3]

def depth2point3D(depth, K):
    height, width = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    points_3d = np.zeros((height, width, 3), dtype=np.float32)

    for v in range(height):
        for u in range(width):
            z = depth[v, u]
            if z <= 0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points_3d[v, u] = [x, y, z]

    return points_3d


def residual_scale(s, matches, depth, K_left, K_right, T_left_right):
    """
    计算对所有匹配像素对的重投影残差 (u_r_obs - u_r_proj, v_r_obs - v_r_proj)
    拼接到一维数组中，供最小二乘优化.

    输入:
      s: 当前的尺度因子 (标量)
      matches: shape=(N,4) -> [[uL, vL, uR, vR], ...]
      K_left, K_right: 3x3 相机内参
      T_left_right: 4x4 外参变换矩阵
    输出:
      residuals: shape=(2*N,) 的一维数组，每对匹配贡献 (dx, dy).
    """
    residuals = []
    fx_L = K_left[0, 0]
    fy_L = K_left[1, 1]
    cx_L = K_left[0, 2]
    cy_L = K_left[1, 2]

    for (uL, vL, uR_obs, vR_obs) in matches:
        d_pred = depth[uL, vL]  # 单目深度
        # 若超出范围或无效, 可做检查:
        if d_pred <= 0:
            continue

        # 左目像素 -> 左目 3D (乘以未知尺度 s)
        X_L = (uL - cx_L) * d_pred * s / fx_L
        Y_L = (vL - cy_L) * d_pred * s / fy_L
        Z_L = d_pred * s

        # 转到右目坐标系
        P_R = transform_point(T_left_right, np.array([X_L, Y_L, Z_L]))

        # 投影到右目图像
        uR_proj, vR_proj = project_point(K_right, P_R)

        # 计算像素偏差
        du = uR_obs - uR_proj
        dv = vR_obs - vR_proj

        residuals.append(du)
        residuals.append(dv)

    return np.array(residuals, dtype=np.float32)

def optimize_scale(depth, matches, K_left, K_right, T_left_right, s_init=1.0):
    """
    使用最小二乘方法, 优化单个尺度因子 s, 使重投影误差最小
    """
    # 定义目标函数
    def f_scale(s):
        return residual_scale(s, matches, depth, K_left, K_right, T_left_right)

    # 使用 scipy.optimize.least_squares
    res = least_squares(
        fun=f_scale,
        x0=np.array([s_init]),
        method='lm',  # 或 'trf', 'dogbox' 等
        max_nfev=100
    )
    return res.x[0], res


# 获取左右图像和左目深度图
def get_images_and_depth(left_image_path, right_image_path, left_depth_path):
    left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
    left_depth = cv2.imread(left_depth_path, cv2.IMREAD_UNCHANGED)  # 假设深度图是单通道16位图像
    return left_image, right_image, left_depth

# 特征匹配
def feature_matching(left_image, right_image):
    # 使用ORB算法进行特征检测和匹配
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_image, None)
    kp2, des2 = orb.detectAndCompute(right_image, None)

    # 使用BFMatcher进行匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    return kp1, kp2, matches

def get_valid_point(kp1, kp2, matches):
    kp1_valid = np.float32([kp1[m.queryIdx].pt for m in matches])
    kp2_valid = np.float32([kp2[m.trainIdx].pt for m in matches])

    return kp1_valid, kp2_valid

def get_valid_point_gt(points, index):
    valid_point = np.array([points[int(y), int(x)] for x, y in index])

    return valid_point


# 计算基础矩阵
def calculate_fundamental_matrix(kp1_valid, kp2_valid):
    F, mask = cv2.findFundamentalMat(kp1_valid, kp2_valid, cv2.FM_8POINT)
    return F, mask

# 三角化计算3D点
def triangulate_points(kp1_valid, kp2_valid, K):
    # 获取左右相机的投影矩阵
    E, _ = cv2.findEssentialMat(kp1_valid, kp2_valid, K)
    _, R, T, _ = cv2.recoverPose(E, kp1_valid, kp2_valid, K)

    # 使用三角化计算3D点
    proj_matrix_left = np.hstack((np.eye(3), np.zeros((3, 1))))  # 左目投影矩阵
    proj_matrix_right = np.hstack((R, T))  # 右目投影矩阵
    points_3d = cv2.triangulatePoints(proj_matrix_left, proj_matrix_right, kp1_valid.T, kp2_valid.T)

    # 将齐次坐标转换为非齐次坐标
    points_3d /= points_3d[3]
    return points_3d[:3].T, kp1_valid, kp2_valid

# 计算重投影误差
def compute_reprojection_error(points_3d, kp1, kp2, K, R, T):
    proj_matrix_left = np.hstack((np.eye(3), np.zeros((3, 1))))  # 左目投影矩阵
    proj_matrix_right = np.hstack((R, T))  # 右目投影矩阵

    # 对 3D 点进行重投影，得到左目和右目上的投影像素
    # 投影到左目图像
    projected_pts_left = K @ points_3d.T # 3xN 矩阵
    projected_pts_left /= projected_pts_left[2]  # 进行透视除法，归一化
    projected_pts_left = projected_pts_left[:2].T  # 转换为 N×2 的形状 (uL, vL)

    # 投影到右目图像
    projected_pts_right = K @ (R @ points_3d.T + T)  # 3xN 矩阵
    projected_pts_right /= projected_pts_right[2]  # 进行透视除法，归一化
    projected_pts_right = projected_pts_right[:2].T  # 转换为 N×2 的形状 (uR, vR)

    # 计算重投影误差，计算左目和右目的投影点与特征点之间的欧氏距离
    error_left = np.sqrt(np.sum((kp1 - projected_pts_left) ** 2, axis=1))
    error_right = np.sqrt(np.sum((kp2 - projected_pts_right) ** 2, axis=1))

    return np.mean(error_left), np.mean(error_right)

# 计算深度的缩放尺度
def compute_scaling_factor(left_depth, points_3d):
    # 假设左目深度图中每个像素的深度值已知
    # 对于每个匹配的点，使用其深度来计算缩放尺度
    depths = left_depth[points_3d[:, 1].astype(int), points_3d[:, 0].astype(int)]
    mean_depth = np.mean(depths)

    # 根据三维点计算的Z值来估计尺度
    mean_z = np.mean(points_3d[:, 2])

    scale = mean_depth / mean_z
    return scale

def get_scale_by_feature_match(file_depth, file_left, file_right, K):
    depth = cv2.imread(file_depth, cv2.IMREAD_UNCHANGED)
    left_image = cv2.imread(file_left, cv2.IMREAD_UNCHANGED)
    right_image = cv2.imread(file_right, cv2.IMREAD_UNCHANGED)
    points =  depth2point3D(depth, K)

    # 假设 baseline = 0.05m, 水平平移
    baseline = 0.05
    T_left_right = np.array([[1, 0, 0, -baseline],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.float32)

    kp1, kp2, matches = feature_matching(left_image, right_image)
    kp1_valid, kp2_valid = get_valid_point(kp1, kp2, matches)

    # 计算基础矩阵
    F, F_mask = calculate_fundamental_matrix(kp1_valid, kp2_valid)

    # 从基础矩阵计算本质矩阵
    E = K.T @ F @ K

    # 对本质矩阵进行分解
    R1, R2, T = cv2.decomposeEssentialMat(E)

    points_3d, kp1_valid, kp2_valid = triangulate_points(kp1_valid, kp2_valid, K)
    points_gt_valid = get_valid_point_gt(points, kp1_valid)


    icp2d = ScaleShiftAnalyzer()
    scale, shift = icp2d.scaling(points_gt_valid, points_3d, local=True)
    pred_aligned = icp2d.align(points_gt_valid, scale, shift)
    diff = np.abs(pred_aligned - points_3d)
    diff /= points_3d
    diff_mean = np.mean(diff)
    print("Mean diff: ", diff_mean)

    # error_left, error_right = compute_reprojection_error(points_3d, kp1_valid, kp2_valid, K, R1, T)
    # print(f"Left Reprojection Error: {error_left}, Right Reprojection Error: {error_right}")

    # 计算深度缩放尺度
    # scale = compute_scaling_factor(depth, points_3d)
    # print(f"Scaling Factor: {scale}")

    return scale

def get_scale_by_lr_consistency(file, intrinsic):
    depth = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    # 假设 baseline = 0.05m, 水平平移
    baseline = 0.05
    T_left_right = np.array([[1, 0, 0, -baseline],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.float32)
    # 假设有 N=5 对匹配点
    matches = np.array([
        [100, 120,  95, 125],
        [200, 150, 195, 152],
        [320, 240, 315, 245],
        [400, 300, 395, 305],
        [260, 130, 255, 128],
    ], dtype=np.uint8)

    # 模拟深度预测 depth_pred 数组(480×640), 这里用常数举例
    H, W = 480, 640
    depth_pred = np.full((H, W), 5.0, dtype=np.float32)  # 假设网络都预测约5米

    # 优化求解尺度
    s_init = 1.0
    s_opt, result = optimize_scale(depth, matches, intrinsic, intrinsic, T_left_right, s_init)
    print(f"Optimal scale = {s_opt:.3f}")
    print(f"Final RMS reprojection error = {np.sqrt(np.mean(result.fun**2)):.3f} pixels")


def main():
    args = GetArgs()
    config  = ConfigLoader()

    if not args.left:
        files = ReadImageList(args.depth)
        for f in files:
            intrinsic = config.set_by_config_yaml(f)
            scale = get_scale_by_lr_consistency(f, intrinsic)
    else:
        files = match_images([args.depth, args.left, args.right])

        for depth, left, right in zip(*files):
            intrinsic = config.set_by_config_yaml(depth)
            scale = get_scale_by_feature_match(depth, left, right, intrinsic)


if __name__ == '__main__':
    main()
