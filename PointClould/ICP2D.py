#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: ICP2D.py
@time: 2025/1/23 15:32
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import numpy as np

def scaling(prediction, target, mask = None):
    if mask is not None:
        prediction = prediction[mask].astype(np.float64)
        target = target[mask].astype(np.float64)
    else:
        prediction = prediction.astype(np.float64)
        target = target.astype(np.float64)

    a_00 = np.sum(prediction * prediction)
    a_01 = np.sum(prediction)
    a_11 = len(prediction)
    b_0 = np.sum(prediction * target)
    b_1 = np.sum(target)

    det = a_00 * a_11 - a_01 * a_01
    if det < 1e-12:
        return 1.0, 0.0

    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det

    return scale, shift

def scaling_point_local(prediction, target, mask=None):
    if mask is not None:
        prediction = prediction[mask].astype(np.float64)
        target = target[mask].astype(np.float64)
    else:
        prediction = prediction.astype(np.float64)
        target = target.astype(np.float64)

    # 将点云从三维转换为二维（对每个坐标轴分别进行计算）
    pred_x, pred_y, pred_z = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    target_x, target_y, target_z = target[:, 0], target[:, 1], target[:, 2]

    scale_x, shift_x = scaling(pred_x, target_x)
    scale_y, shift_y = scaling(pred_y, target_y)
    scale_z, shift_z = scaling(pred_z, target_z)

    return (scale_x, scale_y, scale_z), (shift_x, shift_y, shift_z)


class ScaleShiftAnalyzer:
    def __init__(self):
        self.scales = []
        self.shifts = []

    def scaling(self, prediction, target, mask = None, local = False):
        if local:
            scale, shift = scaling_point_local(prediction, target, mask)
        else:
            scale, shift = scaling(prediction, target, mask)

        self.scales.append(scale)
        self.shifts.append(shift)

        return scale, shift

    def align(self, prediction, scale, shift):
        prediction = prediction.astype(np.float64)
        prediction = prediction * scale + shift
        prediction[prediction < 0] = 0
        prediction[prediction > 65535] = 65535
        return prediction.astype(prediction.dtype)

    def plot_scale_and_shift(self):

        import matplotlib.pyplot as plt

        scale_mean = np.mean(self.scales)
        scale_var = np.var(self.scales)
        shift_mean = np.mean(self.shifts)
        shift_var = np.var(self.shifts)

        print(f"Scale - Mean: {scale_mean}, Variance: {scale_var}")
        print(f"Shift - Mean: {shift_mean}, Variance: {shift_var}")

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.scales, label='Scale')
        plt.xlabel('Image Index')
        plt.ylabel('Scale')
        plt.title('Scale over Images')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.shifts, label='Shift')
        plt.xlabel('Image Index')
        plt.ylabel('Shift')
        plt.title('Shift over Images')
        plt.legend()

        plt.tight_layout()
        plt.show()


def generate_test_points():
    rng = np.random.default_rng(42)
    N = 100
    x = rng.random(N) * 10
    y = 3 * x + 2 + rng.normal(0, 2, size=N)
    return x, y

def main():
    # Generate test points
    x, y = generate_test_points()
    mask = np.ones_like(x, dtype=bool)

    analyzer = ScaleShiftAnalyzer()
    scale, shift = analyzer.scaling(x, y, mask)
    print(f"Computed scale: {scale}, shift: {shift}")

    analyzer.plot_scale_and_shift()

if __name__ == '__main__':
    main()