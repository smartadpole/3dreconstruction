#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: OctoMap.py
@time: 2025/3/18 16:07
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import math
import numpy as np
from collections import deque
from utils.utils import timeit

################################################################################
# 辅助函数：概率与对数几率转换
################################################################################
def probability_to_logodds(p):
    """ 概率 p -> 对数几率 log(p/(1-p)) """
    p = max(min(p, 0.999999), 0.000001)  # 避免 0 或 1 引发数值问题
    return math.log(p / (1.0 - p))

def logodds_to_probability(logodds):
    """ 对数几率 -> 概率 """
    return 1.0 - 1.0/(1.0 + math.exp(logodds))


################################################################################
# OctreeNode: 每个节点新增了 color 与 color_count 用于储存和累积颜色
################################################################################
class OctreeNode:
    """
    升级后的八叉树节点:
    - center: 节点中心坐标 (x, y, z)
    - size: 该节点包围盒的边长
    - depth: 当前节点的层级(根节点=0)
    - log_odds: 占据概率的对数几率 (初始化可为 0 -> p=0.5 代表未知)
    - children: 固定长度8 (None或OctreeNode)
    - is_leaf: 标记是否叶节点
    - last_update: 记录最后一次更新(插入/观测)的时间戳/帧号
    - color: 节点的平均颜色 (R,G,B), float类型, 范围可自定
    - color_count: 累计颜色观测次数，用于做简单平均
    """
    def __init__(self, center, size, depth=0):
        self.center = np.array(center, dtype=float)
        self.size = float(size)
        self.depth = depth

        self.log_odds = 0.0    # 初始未知(0.5概率)
        self.children = [None] * 8
        self.is_leaf = True

        self.last_update = 0  # 动态场景时可用来做衰减

        # 新增颜色字段
        self.color = np.array([0, 0, 0], dtype=float)
        self.color_count = 0

    def __repr__(self):
        return f"<Node center={self.center}, size={self.size}, depth={self.depth}, leaf={self.is_leaf}>"


################################################################################
# OctoMap: 支持概率更新、颜色累积、动态衰减、剪枝、以及点云提取
################################################################################
class OctoMap:
    """
    - 采用对数几率进行概率更新(hit/miss/clamping)
    - 使用多分辨率八叉树, 并可选实现 3D Bresenham 射线投射
    - 支持动态场景(对长时间未被更新的节点做衰减)
    - 可做剪枝合并(若8个子节点占据状态相似)
    - 节点同时存储颜色信息
    """
    def __init__(self,
                 resolution=0.1,
                 max_depth=16,
                 prob_hit=0.7,
                 prob_miss=0.4,
                 clamp_min=0.12,
                 clamp_max=0.97,
                 decay_factor=0.1,
                 decay_time=50,
                 root_size=None):
        self.resolution = resolution
        self.max_depth = max_depth

        # 命中/空闲的对数增量
        self.log_hit = probability_to_logodds(prob_hit)
        self.log_miss = probability_to_logodds(prob_miss)
        self.clamp_min = probability_to_logodds(clamp_min)
        self.clamp_max = probability_to_logodds(clamp_max)

        self.decay_factor = decay_factor
        self.decay_time = decay_time

        if root_size is None:
            root_size = resolution * (2 ** max_depth)
        # 构建根节点, 假定根节点中心在(0,0,0)
        self.root = OctreeNode(center=(0,0,0), size=root_size, depth=0)

        self.global_frame_count = 0  # 用于记录插入次数/帧号

    ############################################################################
    # 1) 主函数: 更新(插入)观测
    ############################################################################
    def insert_pointcloud(self, origin, points, colors, weak_update=False):
        """
        将带有颜色的点云插入OctoMap:
        :param origin: (x, y, z)
        :param points: Nx3 (list 或 np.array)
        :param colors: Nx3 (list 或 np.array), 与 points 对应
        :param weak_update: 是否弱更新
        """
        self.global_frame_count += 1
        factor = 0.2 if weak_update else 1.0

        for (end_pt, color) in zip(points, colors):
            self._cast_and_update(origin, end_pt, factor, color)

        # 可选: 定期衰减 & 剪枝
        if self.global_frame_count % 10 == 0:
            self.apply_decay()
            self.prune()

    def _cast_and_update(self, origin, end_pt, factor, color):
        """
        对单个点执行射线遍历, 中途为 free, 终点为 occupied. 并可更新颜色
        """
        ray_points = self._raycast_cells(origin, end_pt)
        if not ray_points:
            return

        # 中途(忽略最后一个元素)都做 miss
        for cell_center in ray_points[:-1]:
            node = self._update_logodds(cell_center, self.log_miss * factor)
            # 自由区是否更新颜色? 通常认为 free 节点没有固定颜色，
            # 可不做颜色融合，若想记录“背景色”，可自行修改
            # e.g. self._update_color(node, np.array([0,0,0]))

        # 终点
        node = self._update_logodds(ray_points[-1], self.log_hit * factor)
        # 如果给定颜色, 融合到节点
        if color is not None:
            self._update_color(node, color)

    def _update_logodds(self, point, log_delta):
        """
        找到 point 对应的叶节点, 更新 log_odds + log_delta
        并判断是否需细分
        """
        node = self._locate_leaf(self.root, point, 0)
        new_lo = node.log_odds + log_delta
        new_lo = max(min(new_lo, self.clamp_max), self.clamp_min)
        node.log_odds = new_lo
        node.last_update = self.global_frame_count

        # 若节点尺寸大于最小分辨率 & 深度<max_depth, 可细分
        if node.is_leaf and node.depth < self.max_depth:
            if node.size > self.resolution:
                prob = logodds_to_probability(node.log_odds)
                if 0.2 < prob < 0.8:
                    self._subdivide(node)
        return node

    def _update_color(self, node, color):
        """
        将新的观测 color 融合到 node.color 中，做简单平均
        """
        # 如果 node.color_count = 0，说明还没有存过颜色
        ccount = node.color_count
        old_col = node.color
        new_col = (old_col * ccount + color) / (ccount + 1)
        node.color = new_col
        node.color_count += 1

    ############################################################################
    # 2) 细分, 查找 & 射线遍历
    ############################################################################
    def _subdivide(self, node):
        if not node.is_leaf:
            return
        if node.depth >= self.max_depth or node.size <= self.resolution:
            return

        half = node.size / 2.0
        offsets = [
            (-half/2, -half/2, -half/2),
            (-half/2, -half/2,  half/2),
            (-half/2,  half/2, -half/2),
            (-half/2,  half/2,  half/2),
            ( half/2, -half/2, -half/2),
            ( half/2, -half/2,  half/2),
            ( half/2,  half/2, -half/2),
            ( half/2,  half/2,  half/2),
        ]
        for i in range(8):
            child_center = node.center + np.array(offsets[i])
            child = OctreeNode(child_center, half, node.depth + 1)
            # 继承父节点 log_odds / color
            child.log_odds = node.log_odds
            child.last_update = node.last_update
            child.color = np.copy(node.color)
            child.color_count = node.color_count
            node.children[i] = child
        node.is_leaf = False

    def _locate_leaf(self, node, point, depth):
        if node.is_leaf or depth >= self.max_depth:
            return node

        idx = self._child_index(node, point)
        child = node.children[idx]
        if child is None:
            # 理论上不会出现, 因为 subdivide 后 child必然存在
            return node
        return self._locate_leaf(child, point, depth + 1)

    def _child_index(self, node, point):
        idx = 0
        if point[0] >= node.center[0]:
            idx |= 1
        if point[1] >= node.center[1]:
            idx |= 2
        if point[2] >= node.center[2]:
            idx |= 4
        return idx

    def _raycast_cells(self, start, end):
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        vec = end - start
        dist = np.linalg.norm(vec)
        if dist < 1e-9:
            return [start]

        step_size = self.resolution * 0.5
        steps = int(dist / step_size) + 1
        direction = vec / dist
        # Vectorized generation of sample points
        t_values = np.linspace(0, dist, steps)
        ray_points = (start + np.outer(t_values, direction)).tolist()
        ray_points.append(end)
        return ray_points

    ############################################################################
    # 3) 动态场景衰减
    ############################################################################
    @timeit(1)
    def apply_decay(self):
        self._decay_recursive(self.root)

    def _decay_recursive(self, node):
        if node.is_leaf:
            age = self.global_frame_count - node.last_update
            if age > self.decay_time:
                p = logodds_to_probability(node.log_odds)
                if p > 0.5:
                    node.log_odds -= probability_to_logodds(self.decay_factor)
                    node.log_odds = max(node.log_odds, self.clamp_min)
            return
        else:
            for child in node.children:
                if child is not None:
                    self._decay_recursive(child)

    ############################################################################
    # 4) 剪枝合并
    ############################################################################
    @timeit(1)
    def prune(self):
        self._prune_recursive(self.root)

    def _prune_recursive(self, node):
        if node.is_leaf:
            return True

        can_prune = True
        for c in node.children:
            if c is None or not self._prune_recursive(c):
                can_prune = False

        if can_prune:
            # 查看子节点概率是否足够接近
            probs = [logodds_to_probability(c.log_odds) for c in node.children if c is not None]
            if len(probs) < 8:
                return False
            min_p, max_p = min(probs), max(probs)
            if (max_p - min_p) < 0.05:
                # 合并: 取平均
                node.log_odds = sum(c.log_odds for c in node.children if c) / 8.0
                # 合并颜色(可做简单平均)
                sum_col = np.zeros(3, dtype=float)
                sum_count = 0
                for c in node.children:
                    if c is not None:
                        sum_col += c.color * c.color_count
                        sum_count += c.color_count
                if sum_count > 0:
                    node.color = sum_col / sum_count
                    node.color_count = sum_count
                node.is_leaf = True
                node.children = [None]*8
                return True
        return False

    ############################################################################
    # 其它辅助接口
    ############################################################################
    def get_occupancy(self, point):
        leaf = self._locate_leaf(self.root, point, 0)
        return logodds_to_probability(leaf.log_odds)

    def get_color(self, point):
        """
        获取某点所在叶节点的颜色(如果节点未被观测, 可能还是默认[0,0,0])
        """
        leaf = self._locate_leaf(self.root, point, 0)
        return leaf.color

    def get_tree_stats(self):
        """
        简单统计: 计算节点数量, 叶子数量, 平均深度
        """
        nodes_count = 0
        leaf_count = 0
        total_depth = 0
        q = deque([self.root])
        while q:
            n = q.popleft()
            nodes_count += 1
            if n.is_leaf:
                leaf_count += 1
                total_depth += n.depth
            else:
                for c in n.children:
                    if c is not None:
                        q.append(c)
        avg_depth = total_depth / (leaf_count + 1e-9)
        return {
            "total_nodes": nodes_count,
            "leaf_nodes": leaf_count,
            "avg_leaf_depth": avg_depth
        }

    def extractPointCloud(self, occ_threshold=0.5):
        """
        遍历整棵八叉树，提取占据节点和空闲节点的中心点云及颜色。
        :param occ_threshold: 大于该概率即视为“占据”
        :return:
            occupied_points (Nx3),
            occupied_colors (Nx3),
            empty_points   (Mx3),
            empty_colors   (Mx3)
        """
        occupied_pts = []
        occupied_cols = []
        empty_pts = []
        empty_cols = []

        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node.is_leaf:
                p = logodds_to_probability(node.log_odds)
                if p >= occ_threshold:
                    occupied_pts.append(node.center)
                    occupied_cols.append(node.color)
                else:
                    empty_pts.append(node.center)
                    empty_cols.append(node.color)
            else:
                for c in node.children:
                    if c is not None:
                        queue.append(c)

        occupied_pts = np.array(occupied_pts) if occupied_pts else np.empty((0,3))
        occupied_cols = np.array(occupied_cols) if occupied_cols else np.empty((0,3))
        empty_pts = np.array(empty_pts) if empty_pts else np.empty((0,3))
        empty_cols = np.array(empty_cols) if empty_cols else np.empty((0,3))
        return occupied_pts, occupied_cols, empty_pts, empty_cols


################################################################################
# 使用示例
################################################################################
if __name__ == "__main__":
    octomap = OctoMap(
        resolution=0.1,
        max_depth=8,
        prob_hit=0.7,
        prob_miss=0.4,
        clamp_min=0.12,
        clamp_max=0.97,
        decay_factor=0.1,
        decay_time=30
    )

    origin = (0,0,0)

    # 示例: 插入 50 帧，每帧 10 个带颜色的点
    for i in range(50):
        # 在 (1,1,1) 附近随机分布
        pts = []
        cols = []
        for _ in range(10):
            px = 1 + 0.1*np.random.randn()
            py = 1 + 0.1*np.random.randn()
            pz = 1 + 0.1*np.random.randn()
            # 颜色也随机一下, 例如 RGB in [0,255]
            cr = np.random.randint(0, 256)
            cg = np.random.randint(0, 256)
            cb = np.random.randint(0, 256)
            pts.append((px, py, pz))
            cols.append((cr, cg, cb))

        # 插入带颜色的点云
        octomap.insert_pointcloud_with_color(origin, pts, cols, weak_update=(i<10))

    # 查询某点的占据概率和颜色
    test_pt = (1,1,1)
    p = octomap.get_occupancy(test_pt)
    c = octomap.get_color(test_pt)
    print(f"Occupancy prob at {test_pt} = {p:.3f}, color = {c}")

    # 输出统计
    stats = octomap.get_tree_stats()
    print("Tree stats:", stats)

    # 提取点云
    occ_pts, occ_cols, emp_pts, emp_cols = octomap.extractPointCloud(occ_threshold=0.5)
    print(f"Occupied pointcloud shape: {occ_pts.shape}, {occ_cols.shape}")
    print(f"Empty    pointcloud shape: {emp_pts.shape}, {emp_cols.shape}")

    # 可进一步使用 open3d/pcl 做可视化
    # ...
