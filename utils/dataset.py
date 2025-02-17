#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: dataset.py
@time: 2025/2/5 11:17
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import numpy as np


CONFIG_FILE = ['config.yaml', 'MODULE.yaml', 'MoudleParam.yaml', 'left.yaml', 'right.yaml']


def generate_sample_index(num_frames, skip_frames, sequence_length):
    sample_index_list = []
    k = skip_frames
    demi_length = (sequence_length - 1) // 2
    shifts = list(range(-demi_length * k,
                        demi_length * k + 1, k))
    shifts.pop(demi_length)

    if num_frames > sequence_length:
        for i in range(demi_length * k, num_frames - demi_length * k):
            sample_index = {'tgt_idx': i, 'ref_idx': []}
            for j in shifts:
                sample_index['ref_idx'].append(i + j)
            sample_index_list.append(sample_index)

    return sample_index_list


import os
import yaml

CONFIG_FILE = ['config.yaml', 'MODULE.yaml', 'MoudleParam.yaml', 'left.yaml', 'right.yaml']


def get_config_file(path):
    for file_name in CONFIG_FILE:
        file = os.path.join(path, file_name)
        if os.path.exists(file):
            if ('left' in path and 'right' not in file) or (
                    'right' in path and 'left' not in file):
                break
            else:
                break
    return file


class ConfigLoader:
    def __init__(self):
        self.K_dict = {}

    def set_by_config_yaml(self, folder):
        K = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float32)
        image_path_length = len(folder.split('/'))
        for index in range(1, image_path_length):
            config_file = "/" + os.path.join(*(folder.split('/')[:-1 * index]))
            config_file = get_config_file(config_file)
            if os.path.exists(config_file):
                break
        if config_file in self.K_dict:
            K = self.K_dict[config_file]
        else:
            try:
                with open(config_file, 'r') as f:
                    for line in f:
                        if 'fx' in line:
                            K[0][0] = float(line.split(':')[-1].strip())
                        elif 'cx' in line:
                            K[0][2] = float(line.split(':')[-1].strip())
                        elif 'fy' in line:
                            K[1][1] = float(line.split(':')[-1].strip())
                        elif 'cy' in line:
                            K[1][2] = float(line.split(':')[-1].strip())
                        self.K_dict[config_file] = K
            except:
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f)
                    K = np.array(data['K']).reshape(3, 3)
                    # K[0][0] = data['K'][0]  # fx
                    # K[0][2] = data['K'][2]  # cx
                    # K[1][1] = data['K'][4]  # fy
                    # K[1][2] = data['K'][5]  # fy
                    self.K_dict[config_file] = K
        return K