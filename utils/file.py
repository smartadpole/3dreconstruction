#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: utils.py
@time: 2020/12/4 下午12:28
@desc:
'''
import os
import re

__all__ = ["FILE_SUFFIX", "Walk", "MkdirSimple", "WriteTxt", "WalkImage", "GetImages", 'ReadImageList', 'match_images']

FILE_SUFFIX = ['jpg', 'png', 'jpeg', 'bmp', 'tiff']


def Walk(path, suffix:tuple):
    file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path, followlinks=True) for f in filenames if f.endswith(suffix)]

    try:
        file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))
    except:
        file_list.sort()

    return file_list

def WalkImage(path):
    return Walk(path, tuple(FILE_SUFFIX))

def ReadImageList(image_path):
    """Get a list of image paths from a directory or a single image path."""
    if not image_path:
        return []
    if os.path.isfile(image_path):
        if image_path.endswith(tuple(FILE_SUFFIX)):
            return [image_path]
        else:
            image_list = []
            with open(image_path, "r") as file:
                image_list = file.readlines()
                image_list = [f.strip() for f in image_list]
            return image_list
    elif os.path.isdir(image_path):
        return WalkImage(image_path)
    else:
        raise Exception("Cannot find image_path: {}".format(image_path))

def MkdirSimple(path):
    path_current = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if path_current not in ["", "./", ".\\"]:
        os.makedirs(path_current, exist_ok=True)

def WriteTxt(txt, path, encoding):
    MkdirSimple(path)
    with open(path, encoding) as out:
        out.write(txt)


def GetImages(path):
    if os.path.isfile(path):
        # Only testing on a single image
        paths = [path]
        root_len = len(os.path.dirname(paths).rstrip('/'))
    elif os.path.isdir(path):
        # Searching folder for images
        paths = WalkImage(path)
        root_len = len(path.rstrip('/'))
    else:
        raise Exception("Can not find path: {}".format(path))

    return paths, root_len

def match_images(paths:list):
    lists = [ReadImageList(path) for path in paths]

    common_files = None
    for i, files in enumerate(lists):
        files = [os.path.relpath(f, start=paths[i]) for f in files]
        if common_files is None:
            common_files = set(files)
        else:
            common_files.intersection_update(files)

    common_files = list(common_files)
    try:
        common_files.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))
    except:
        common_files.sort()
    common_files = list(common_files)
    matched_lists = []
    for root in paths:
        matched_lists.append([os.path.join(root, file) for file in common_files])

    return matched_lists