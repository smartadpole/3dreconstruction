#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: utils.py
@time: 2025/2/17 10:04
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import time
from functools import wraps

__all__ = ['timeit']

# Decorator function to measure the time taken by a function
def timeit(time_len):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            wrapper.times.append(elapsed_time)
            if len(wrapper.times) % time_len == 0:
                average_time = sum(wrapper.times[-time_len:]) / time_len
                print(f"Average time for last {time_len} frames in {func.__name__}: {average_time * 1000:.1f} ms")
            return result
        wrapper.times = []
        return wrapper
    return decorator