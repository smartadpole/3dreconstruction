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


def copy_point_cloud(pcd):
    """
    Open3D 的 .transform() 是原地操作，为防止覆盖前的点云数据，需要深拷贝。
    """
    return o3d.geometry.PointCloud(pcd)


def array_to_pcd(points_xyz: np.ndarray):
    """
    将 Nx3 的 NumPy 点云转换为 Open3D 的 PointCloud 对象。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)  # 设置点的坐标
    return pcd

class ICPRegistration:
    """
    用于将一系列点云按照帧顺序做相邻ICP配准，并累积到全局坐标系中。
    默认的ICP为 point-to-plane 模式，适合高密度或有法线的信息的点云。
    """

    def __init__(self, max_corr_distance=0.2, estimation_method='point_to_plane'):
        """
        :param max_corr_distance: ICP搜索对应点的最大距离(单位:米)，需根据场景大小调整
        :param estimation_method: ICP的估计方式，可选 'point_to_point' 或 'point_to_plane'
        """
        self.max_corr_distance = max_corr_distance
        self.estimation_method = estimation_method

        self.global_poses = []
        self.accumulated_pcd = o3d.geometry.PointCloud()
        self.current_world_pose = np.eye(4)
        self.prev_pcd_world = None

    def icp_registration(self, pcd, init_trans=np.eye(4)):
        """
        执行ICP，将 source_pcd 对齐到 target_pcd 的坐标系。
        :param source_pcd: open3d.geometry.PointCloud
        :param target_pcd: open3d.geometry.PointCloud
        :param init_trans: 初始变换猜测，默认单位阵
        :return: 4x4 变换矩阵 T_st (把source坐标系映射到target坐标系)
        """

        if not isinstance(pcd, o3d.geometry.PointCloud):
            pcd = array_to_pcd(pcd)

        pcd.estimate_normals()

        if self.prev_pcd_world is None:
            self.prev_pcd_world = copy_point_cloud(pcd)
            return np.eye(4)

        if self.estimation_method == 'point_to_plane':
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        # ICP 配准
        icp_result = o3d.pipelines.registration.registration_icp(
            pcd, self.prev_pcd_world,
            self.max_corr_distance,
            init_trans,
            estimator
        )
        return icp_result.transformation

    def register_frame(self, current_pcd):
        """
        注册(配准)当前帧点云到全局坐标系。
        :param current_pcd: open3d.geometry.PointCloud, 尚未变换到世界坐标系
        :return: 当前帧的4x4全局位姿矩阵
        """
        if self.prev_pcd_world is None:
            self.current_world_pose = np.eye(4)
        else:
            T_st = self.icp_registration(current_pcd, self.prev_pcd_world, init_trans=np.eye(4))
            self.current_world_pose = np.dot(self.current_world_pose, T_st)

        current_pcd_world = copy_point_cloud(current_pcd)
        current_pcd_world.transform(self.current_world_pose)

        self.accumulated_pcd += current_pcd_world
        self.prev_pcd_world = current_pcd_world
        self.global_poses.append(self.current_world_pose.copy())

        return self.current_world_pose

    def get_current_pose_matrix(self):
        """
        获取当前帧(最后一帧)的世界位姿(4x4矩阵)。
        """
        return self.current_world_pose

    def get_global_poses(self):
        """
        返回所有帧在世界坐标系下的位姿矩阵列表
        """
        return self.global_poses

    def get_accumulated_pointcloud(self):
        """
        返回累积后的全局点云
        """
        return self.accumulated_pcd
