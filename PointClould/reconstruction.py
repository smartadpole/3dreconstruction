#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: reconstruction.py
@time: 2025/2/19 10:03
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse

import cv2
import numpy as np
from utils.utils import timeit
from PointClould.ICP2D import ScaleShiftAnalyzer
from utils.dataset import ConfigLoader
from utils.file import match_images, ReadImageList
from tqdm.contrib import tzip
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PointClould.ICP import ICPRegistration


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--depth", type=str, required=True, help="depth image dir or file path or list of depth image files")
    parser.add_argument("--pose", type=str, help="pose dir or file path or list of pose files")
    parser.add_argument("--extrinsic", type=str, required=True, help="extrinsic pose file")
    parser.add_argument("--left", type=str, help="left image dir or file path or list of depth image files")

    args = parser.parse_args()
    return args

# 假设存在的配置
MIN = 0
VALID_DISTANCE = 300  # 7m


def to_rotation(position, orientation):
    # position 是平移向量
    translation = np.array([position['x'], position['y'], position['z']])

    # orientation 是四元数 [x, y, z, w]
    quat = [orientation['x'], orientation['y'], orientation['z'], orientation['w']]

    # 创建四元数对象
    rotation = R.from_quat(quat)

    # 获取旋转矩阵
    rotation_matrix = rotation.as_matrix()  # 3x3 旋转矩阵

    return rotation_matrix, translation

def load_pose(pose_file):
    poses = {}

    with open(pose_file, 'r') as file:
        data = yaml.safe_load_all(file)

        for frame in data:
            if not frame:
                continue
            pose_data = frame['pose']['pose']

            position = pose_data['position']
            orientation = pose_data['orientation']
            secs = frame['header']['stamp']['secs']
            nsecs = frame['header']['stamp']['nsecs']
            nsecs = f"{nsecs:09d}"[:-3]

            pose_key = f"{secs}{nsecs}"
            pose = {
                'position': position,  # 包含x, y, z
                'orientation': orientation  # 包含x, y, z, w
            }
            poses[pose_key] = pose

    return poses


def get_pose(poses, file):
    timestamp = os.path.splitext(os.path.basename(file))[0]
    timestamp = timestamp[:-3]

    if timestamp not in poses:
        return None, None

    p = poses[str(timestamp)]
    rotation, translation = to_rotation(p['position'], p['orientation'])

    return rotation, translation


def load_extrinsic(file):
    with open(file, 'r') as file:
        data = yaml.safe_load(file)

    # 提取 IMU 和相机相关的参数
    imu = data.get('imu', None)
    num_of_cam = data.get('num_of_cam', None)

    # 提取 IMU 与每个相机之间的外参矩阵
    body_T_cam = {}
    for i in range(num_of_cam):
        cam_key = f'body_T_cam{i}'
        if cam_key in data:
            body_T_cam[i] = np.array(data[cam_key]['data']).reshape(4, 4)

    # 提取其他配置参数
    imu_topic = data.get('imu_topic', None)
    image_topics = [data.get(f'image{i}_topic', None) for i in range(num_of_cam)]
    output_path = data.get('output_path', None)

    # 返回加载的配置字典
    return {
        'imu': imu,
        'num_of_cam': num_of_cam,
        'body_T_cam': body_T_cam,
        'imu_topic': imu_topic,
        'image_topics': image_topics,
        'output_path': output_path
    }


def get_camera_pose(imu_rotation_matrix, imu_translation, pose_imu2camera):
    imu_pose = np.eye(4)
    imu_pose[:3, :3] = imu_rotation_matrix
    imu_pose[:3, 3] = imu_translation

    camera_pose = np.dot(pose_imu2camera, imu_pose)
    camera_rotation = camera_pose[:3, :3]
    camera_translation = camera_pose[:3, 3]
    # camera_rotation = R.from_matrix(camera_rotation)

    return camera_rotation, camera_translation


def depth2point_cloud(dpeth, K, image=None):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    height, width = dpeth.shape
    if image is not None:
        height, width = min(dpeth.shape[0], image.shape[0]), min(dpeth.shape[1], image.shape[1])
    point_cloud = []
    colors = []

    for v in range(height):
        for u in range(width):
            Z = dpeth[v, u]
            if Z < 0 :
                continue

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            point_cloud.append([X, Y, Z])

            if image is not None:
                color = image[v, u] / 255.0
                colors.append(color)

    return np.array(point_cloud), np.array(colors)

# 将深度图转换为3D点云
def depth2point_cloud_matrix(depth, K):
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

def preprocess(depth):
    # depth = cv2.medianBlur(depth, 5)

    valid_mask = (depth > MIN) & (depth < 65535)
    depth[~valid_mask] = 0

    depth = (depth / 65535.0 * 2000).astype(np.uint16) # max valid distance is 20m
    depth[depth > VALID_DISTANCE] = 0

    return depth

# 将3D点转换为深度图
def point3D2depth(points_3d):
    height, width, _ = points_3d.shape
    depth = np.zeros((height, width), dtype=np.float32)

    for v in range(height):
        for u in range(width):
            x, y, z = points_3d[v, u]
            if z <= 0:
                continue
            depth[v, u] = z

    return depth

# 读取深度图和左图像
def get_images_and_depth(left_image_path, left_depth_path):
    left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
    left_depth = cv2.imread(left_depth_path, cv2.IMREAD_UNCHANGED)  # 假设深度图是单通道16位图像
    return left_image, left_depth

# 点云融合处理
def process_point_cloud(depth, K, left_image=None):
    depth = preprocess(depth)

    # 使用ICP进行点云对齐等操作
    # ICP = ScaleShiftAnalyzer()
    # scale, shift = ICP.scaling(points_3d, points_3d)
    # points_3d_aligned = ICP.align(points_3d, scale, shift)

    point_cloud, colors = depth2point_cloud(depth, K, left_image)

    return point_cloud, colors

# 使用位姿进行坐标变换
def transform_point_cloud(point_cloud, rotation, translation):
    # 对点云进行坐标变换
    point_cloud_transformed = np.dot(point_cloud, rotation.T) + translation.T
    return point_cloud_transformed


def visualize_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    for i, cloud in enumerate(points):
        pcd.points.extend(o3d.utility.Vector3dVector(cloud))
        if len(colors) == len(points):
            pcd.colors.extend(o3d.utility.Vector3dVector(colors[i]))

    o3d.visualization.draw_geometries([pcd])
    return

# 主函数处理整个过程
def main():
    args = GetArgs()
    config  = ConfigLoader()
    point_clouds = []
    colors = []

    poses = load_pose(args.pose) if args.pose else None
    extrinsic = load_extrinsic(args.extrinsic)

    if not args.left:
        files = ReadImageList(args.depth)
        files = [files, [None, ]*len(files)]
    else:
        files = match_images([args.depth, args.left])

    root_len = len(args.depth.rstrip('/'))

    ICP = ICPRegistration(max_corr_distance=10)

    for idx, (depth_file, left_file) in enumerate(tzip(*files)):
        intrinsic = config.set_by_config_yaml(depth_file)

        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        left = cv2.imread(left_file, cv2.IMREAD_UNCHANGED) if left_file else None
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB) if left is not None else None
        pcd, color = process_point_cloud(depth, intrinsic, left)

        if poses:
            rotation, translation = get_pose(poses, depth_file)
        else:
            current_world_pose = ICP.icp_registration(pcd)
            rotation = current_world_pose[:3, :3]  # 3x3
            translation = current_world_pose[:3, 3]  # 3x1

        if rotation is None:
            continue
        rotation_camera, translation_camera = get_camera_pose(rotation, translation, extrinsic['body_T_cam'][0])

        pcd_transformed = transform_point_cloud(pcd, rotation_camera, translation_camera)

        point_clouds.append(pcd_transformed)
        if left_file is not None:
            colors.append(color)

        if len(point_clouds) % 10 == 0:
            visualize_point_cloud(point_clouds, colors)

    visualize_point_cloud(point_clouds, colors)

    output_dir = os.path.dirname(args.depth) if os.path.isfile(args.depth) else args.depth
    output_file = os.path.join(output_dir, "world.pcl")


if __name__ == '__main__':
    main()

