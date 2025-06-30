#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: reconstruction.py
@time: 2025/2/19 10:03
@desc: 改进的点云重建，专注于精确配准和配准后滤波
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse
import cv2
import numpy as np
from utils.utils import timeit
from utils.dataset import ConfigLoader
from utils.file import match_images, ReadImageList
from tqdm.contrib import tzip
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PointClould.ICP import array_to_pcd


def GetArgs():
    parser = argparse.ArgumentParser(description="精确配准的点云重建",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--depth", type=str, required=True, help="深度图目录或文件路径")
    parser.add_argument("--pose", type=str, help="位姿文件路径")
    parser.add_argument("--extrinsic", type=str, required=True, help="外参配置文件")
    parser.add_argument("--left", type=str, help="RGB图像目录或文件路径")
    parser.add_argument("--output", type=str, default="merged_pointcloud.ply", help="输出点云文件路径")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="体素下采样大小(m)")
    parser.add_argument("--max_depth", type=float, default=20.0, help="最大有效深度(m)")
    parser.add_argument("--min_depth", type=float, default=0.1, help="最小有效深度(m)")
    parser.add_argument("--icp_threshold", type=float, default=0.02, help="ICP配准阈值")
    parser.add_argument("--filter_radius", type=float, default=0.05, help="半径滤波半径")
    
    args = parser.parse_args()
    return args

# 改进的配置参数
MIN_DEPTH = 0.1  # m
MAX_DEPTH = 20.0  # m
VOXEL_SIZE = 0.02  # m
ICP_THRESHOLD = 0.02  # m
FILTER_RADIUS = 0.05  # m
VALID_DISTANCE = 3  # m
STEP_SHOW = 10  # 每10帧显示一次进度


def to_rotation(position, orientation):
    """将位置和四元数转换为旋转矩阵和平移向量"""
    translation = np.array([position['x'], position['y'], position['z']])
    quat = [orientation['x'], orientation['y'], orientation['z'], orientation['w']]
    rotation = R.from_quat(quat)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix, translation


def load_pose(pose_file):
    """加载位姿文件"""
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
                'position': position,
                'orientation': orientation
            }
            poses[pose_key] = pose
    return poses


def get_pose(poses, file):
    """根据文件名获取对应的位姿"""
    timestamp = os.path.splitext(os.path.basename(file))[0]
    timestamp = timestamp[:-3]
    
    if timestamp not in poses:
        return None, None
    
    p = poses[str(timestamp)]
    rotation, translation = to_rotation(p['position'], p['orientation'])
    return rotation, translation


def load_extrinsic(file):
    """加载外参配置"""
    with open(file, 'r') as file:
        data = yaml.safe_load(file)
    
    num_of_cam = data.get('num_of_cam', None)
    body_T_cam = {}
    for i in range(num_of_cam):
        cam_key = f'body_T_cam{i}'
        if cam_key in data:
            body_T_cam[i] = np.array(data[cam_key]['data']).reshape(4, 4)
    
    return {
        'num_of_cam': num_of_cam,
        'body_T_cam': body_T_cam
    }


def get_camera_pose(imu_rotation_matrix, imu_translation, body_T_cam):
    """计算相机在世界坐标系下的位姿"""
    imu_pose = np.eye(4)
    imu_pose[:3, :3] = imu_rotation_matrix
    imu_pose[:3, 3] = imu_translation
    
    camera_pose = np.dot(body_T_cam, imu_pose)
    camera_rotation = camera_pose[:3, :3]
    camera_translation = camera_pose[:3, 3]
    
    return camera_rotation, camera_translation


def enhanced_preprocess(depth, kernel_size=5):
    """增强的深度图预处理"""
    # 确保深度图是合适的格式
    if depth.dtype != np.uint8 and depth.dtype != np.float32:
        # 将16位深度图转换为32位浮点数
        depth = depth.astype(np.float32)
    
    # 1. 中值滤波去除椒盐噪声
    depth_filtered = cv2.medianBlur(depth, kernel_size)
    
    # 2. 双边滤波保持边缘的同时平滑噪声（需要8位或32位格式）
    if depth_filtered.dtype == np.uint16:
        # 将16位转换为8位进行双边滤波
        depth_normalized = (depth_filtered / 256).astype(np.uint8)
        depth_filtered = cv2.bilateralFilter(depth_normalized, 9, 75, 75)
        # 转换回原始范围
        depth_filtered = (depth_filtered.astype(np.float32) * 256).astype(np.uint16)
    else:
        depth_filtered = cv2.bilateralFilter(depth_filtered, 9, 75, 75)
    
    # 3. 深度范围过滤
    depth_filtered = (depth_filtered / 65535.0 * 2000).astype(np.uint16) # max valid distance is 20m
    valid_mask = (depth_filtered > MIN_DEPTH) & (depth_filtered < MAX_DEPTH)
    depth_filtered[~valid_mask] = 0

    depth_filtered[depth_filtered > VALID_DISTANCE] = 0
    
    # 4. 形态学操作去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    depth_filtered = cv2.morphologyEx(depth_filtered, cv2.MORPH_OPEN, kernel)
    
    return depth_filtered


def depth2point_cloud(depth, K, image=None):
    """将深度图转换为点云"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    height, width = depth.shape
    
    if image is not None:
        height, width = min(depth.shape[0], image.shape[0]), min(depth.shape[1], image.shape[1])
    
    point_cloud = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            Z = depth[v, u] # m
            if Z <= 0 or Z < MIN_DEPTH or Z > MAX_DEPTH:
                continue
            
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            point_cloud.append([X, Y, Z])
            
            if image is not None:
                color = image[v, u] / 255.0
                colors.append(color)
    
    return np.array(point_cloud), np.array(colors)


def create_point_cloud(points, colors=None):
    """创建Open3D点云对象"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pre_filter_point_cloud(pcd, voxel_size=0.02):
    """配准前的点云预处理"""
    if not isinstance(pcd, o3d.geometry.PointCloud):
        pcd = create_point_cloud(pcd)
    
    # 1. 统计滤波去除明显离群点
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 2. 体素下采样减少计算量
    pcd_downsampled = pcd_filtered.voxel_down_sample(voxel_size=voxel_size)
    
    # 3. 估计法向量用于ICP配准
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    return pcd_downsampled


def precise_icp_registration(source_pcd, target_pcd, initial_transform=np.eye(4), max_iterations=100):
    """精确的ICP配准"""
    # 1. 点对点ICP（粗配准）
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, ICP_THRESHOLD * 2, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    # 2. 点对平面ICP（精配准）
    icp_fine = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, ICP_THRESHOLD, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    return icp_fine


def post_registration_filter(merged_pcd, radius=0.05, min_neighbors=10):
    """配准后的点云滤波"""
    # 1. 半径滤波去除配准后的离群点
    pcd_filtered, _ = merged_pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    
    # 2. 统计滤波进一步清理
    pcd_filtered, _ = pcd_filtered.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 3. 体素下采样统一密度
    pcd_downsampled = pcd_filtered.voxel_down_sample(voxel_size=VOXEL_SIZE)
    
    # 4. 重新估计法向量
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    return pcd_downsampled


def process_frame(depth, K, left_image=None):
    """处理单帧数据"""
    # 增强预处理
    depth_processed = enhanced_preprocess(depth)
    
    # 转换为点云
    point_cloud, colors = depth2point_cloud(depth_processed, K, left_image)
    
    if len(point_cloud) == 0:
        return None
    
    # 创建点云对象
    pcd = create_point_cloud(point_cloud, colors)
    
    # 配准前预处理
    pcd_processed = pre_filter_point_cloud(pcd, VOXEL_SIZE)
    
    return pcd_processed


def main():
    args = GetArgs()
    config = ConfigLoader()
    
    # 加载位姿和外参
    poses = load_pose(args.pose) if args.pose else None
    extrinsic = load_extrinsic(args.extrinsic)
    
    # 获取相机外参矩阵（假设使用第一个相机）
    body_T_cam = extrinsic['body_T_cam'].get(0, np.eye(4))
    
    # 准备文件列表
    if not args.left:
        files = ReadImageList(args.depth)
        files = [files, [None] * len(files)]
    else:
        files = match_images([args.depth, args.left])
    
    print(f"开始处理 {len(files[0])} 帧数据...")
    
    # 存储配准后的点云
    registered_point_clouds = []
    current_world_pose = np.eye(4)
    
    for idx, (depth_file, left_file) in enumerate(tzip(*files)):
        print(f"处理第 {idx + 1}/{len(files[0])} 帧")

        if (idx + 1) % STEP_SHOW == 0 and len(registered_point_clouds) > 1:
            print(f"进行中间滤波，当前点云数量: {len(registered_point_clouds)}")

            merged_pcd = o3d.geometry.PointCloud()
            for pcd in registered_point_clouds:
                merged_pcd += pcd

            filtered_pcd = post_registration_filter(merged_pcd, args.filter_radius)
            registered_point_clouds = [filtered_pcd]

            if len(filtered_pcd.points) > 0:
                o3d.visualization.draw_geometries(registered_point_clouds)
        
        # 获取相机内参
        intrinsic = config.set_by_config_yaml(depth_file)
        
        # 读取图像
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        left = cv2.imread(left_file, cv2.IMREAD_COLOR) if left_file else None
        
        if depth is None:
            print(f"警告：无法读取深度图 {depth_file}")
            continue
        
        if left is not None:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            # 确保尺寸一致
            if depth.shape[:2] != left.shape[:2]:
                depth = cv2.resize(depth, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 处理当前帧
        current_pcd = process_frame(depth, intrinsic, left)
        if current_pcd is None or len(current_pcd.points) == 0:
            print(f"警告：第 {idx + 1} 帧点云为空，跳过")
            continue
        
        # 获取位姿并进行配准
        if poses:
            rotation, translation = get_pose(poses, depth_file)
            if rotation is None or translation is None:
                print(f" Warning: No pose found, skipping...")
                continue
            
            # 应用外参变换
            camera_rotation, camera_translation = get_camera_pose(rotation, translation, body_T_cam)
            
            # 计算世界坐标系下的位姿
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = camera_rotation
            camera_pose[:3, 3] = camera_translation
            
            # 变换点云到世界坐标系
            current_pcd.transform(camera_pose)
            current_world_pose = camera_pose
        else:
            # 如果没有提供位姿，使用ICP进行配准
            if len(registered_point_clouds) > 0:
                # 使用上一帧作为目标进行配准
                target_pcd = registered_point_clouds[-1]
                
                # 精确ICP配准
                icp_result = precise_icp_registration(
                    current_pcd, target_pcd, 
                    initial_transform=current_world_pose
                )
                
                # 更新世界位姿
                current_world_pose = np.dot(current_world_pose, icp_result.transformation)
                
                # 应用变换
                current_pcd.transform(icp_result.transformation)
                
                print(f"ICP配准得分: {icp_result.fitness}")

        registered_point_clouds.append(current_pcd)

    
    # 最终合并和滤波
    print("进行最终点云合并和滤波...")
    if len(registered_point_clouds) > 0:
        # 合并所有点云
        final_pcd = o3d.geometry.PointCloud()
        for pcd in registered_point_clouds:
            final_pcd += pcd
        
        # 最终配准后滤波
        final_pcd = post_registration_filter(final_pcd, args.filter_radius)
        
        # 保存结果
        output_path = args.output
        o3d.io.write_point_cloud(output_path, final_pcd)
        print(f"点云已保存到: {output_path}")
        print(f"最终点云包含 {len(final_pcd.points)} 个点")
        
        # 可视化
        o3d.visualization.draw_geometries([final_pcd])
    else:
        print("错误：没有生成有效的点云")


if __name__ == '__main__':
    main()

