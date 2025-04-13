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
from PointClould.ICP import ICPRegistration, array_to_pcd
from PointClould.OctoMap import OctoMap


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
VALID_DISTANCE = 300  # m
RESOLUTION = 5  # cm


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

def radius_filter(pcd_o3d, radius=0.05, min_neighbors=5):
    cl, index = pcd_o3d.remove_radius_outlier(nb_points=min_neighbors, radius=radius)

    return cl, index

def statistical_filter(pcd_o3d, nb_neighbors=20, std_ratio=2.0):
    """
    用统计滤波移除离群点，返回滤波后点云和被移除点索引。
    """
    if not isinstance(pcd_o3d, o3d.geometry.PointCloud):
        pcd_o3d = array_to_pcd(pcd_o3d)
    cl, index = pcd_o3d.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    return cl, index

def process_point_cloud(depth, K, left_image=None):
    depth = preprocess(depth)

    # 使用ICP进行点云对齐等操作
    # ICP = ScaleShiftAnalyzer()
    # scale, shift = ICP.scaling(points_3d, points_3d)
    # points_3d_aligned = ICP.align(points_3d, scale, shift)

    point_cloud, colors = depth2point_cloud(depth, K, left_image)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_filter, index = statistical_filter(pcd)
    pcd_down = pcd_filter.voxel_down_sample(voxel_size=RESOLUTION)

    return pcd_down

# 使用位姿进行坐标变换
def transform_point_cloud(point_cloud, rotation, translation):
    # 对点云进行坐标变换
    point_cloud_transformed = np.dot(point_cloud, rotation.T) + translation.T
    return point_cloud_transformed

def visualize_pcd_by_color(points, colors):
    pcd = o3d.geometry.PointCloud()
    for i, cloud in enumerate(points):
        pcd.points.extend(o3d.utility.Vector3dVector(cloud))
        if len(colors) == len(points):
            pcd.colors.extend(o3d.utility.Vector3dVector(colors[i]))

    o3d.visualization.draw_geometries([pcd])
    return

def visualize_pcd(points):
    pcd = o3d.geometry.PointCloud()
    for i, cloud in enumerate(points):
        pcd += cloud
    o3d.visualization.draw_geometries([pcd])
    return

# 主函数处理整个过程
def main():
    args = GetArgs()
    config  = ConfigLoader()
    point_clouds = []
    colors = []
    occupancy = []
    colors_occ = []

    poses = load_pose(args.pose) if args.pose else None
    extrinsic = load_extrinsic(args.extrinsic)

    if not args.left:
        files = ReadImageList(args.depth)
        files = [files, [None, ]*len(files)]
    else:
        files = match_images([args.depth, args.left])

    root_len = len(args.depth.rstrip('/'))

    # 初始化TSDF体积
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,  # 更小的体素大小以提高精度
        sdf_trunc=0.04,    # 更小的截断距离
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # 用于存储上一帧的位姿
    last_pose = None
    last_pcd = None

    for idx, (depth_file, left_file) in enumerate(tzip(*files)):
        print(f"Processing frame {idx + 1}/{len(files[0])}")
        
        intrinsic = config.set_by_config_yaml(depth_file)
        
        # 读取深度图和RGB图
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        left = cv2.imread(left_file, cv2.IMREAD_COLOR) if left_file else None
        if left is not None:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        
        # 确保深度图和RGB图大小一致
        if left is not None and depth.shape[:2] != left.shape[:2]:
            depth = cv2.resize(depth, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 预处理深度图
        depth = preprocess(depth)
        
        # 创建RGBD图像
        depth_o3d = o3d.geometry.Image(depth)
        color_o3d = o3d.geometry.Image(left) if left is not None else None
        
        if color_o3d is not None:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1000.0,  # 调整深度缩放因子
                depth_trunc=3.0,     # 设置最大深度截断
                convert_rgb_to_intensity=False
            )
        else:
            rgbd_image = o3d.geometry.RGBDImage.create_from_depth_and_intensity(
                depth_o3d,
                depth_scale=1000.0,
                depth_trunc=3.0
            )

        # 获取相机位姿
        if poses:
            rotation, translation = get_pose(poses, depth_file)
            if rotation is None or translation is None:
                print(f"Warning: No valid pose found for frame {idx}, skipping...")
                continue
            current_world_pose = np.eye(4)
            current_world_pose[:3, :3] = rotation
            current_world_pose[:3, 3] = translation
        else:
            # 如果没有提供位姿，使用ICP进行配准
            if last_pcd is None:
                last_pcd = process_point_cloud(depth, intrinsic, left)
                last_pose = np.eye(4)
                continue
            
            current_pcd = process_point_cloud(depth, intrinsic, left)
            icp_result = o3d.pipelines.registration.registration_icp(
                current_pcd, last_pcd, 0.02,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            current_world_pose = np.dot(last_pose, icp_result.transformation)
            last_pcd = current_pcd
            last_pose = current_world_pose

        # 获取相机内参
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=depth.shape[1], height=depth.shape[0],
            fx=intrinsic[0, 0], fy=intrinsic[1, 1],
            cx=intrinsic[0, 2], cy=intrinsic[1, 2]
        )

        # 将当前帧集成到TSDF体积中
        volume.integrate(rgbd_image, intrinsics, current_world_pose)

        # 每处理一定数量的帧后显示中间结果
        if (idx + 1) % 10 == 0:
            mesh = volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])

    # 提取最终的网格
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # 保存重建结果
    output_dir = os.path.dirname(args.depth) if os.path.isfile(args.depth) else args.depth
    output_file = os.path.join(output_dir, "reconstructed_mesh.ply")
    o3d.io.write_triangle_mesh(output_file, mesh)
    
    # 显示最终结果
    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    main()

