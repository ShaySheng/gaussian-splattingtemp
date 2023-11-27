#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 导入 errno 模块中的 EEXIST 常量，用于处理文件存在错误
from errno import EEXIST
# 从 os 模块导入 makedirs 和 path 函数，用于创建目录和处理路径
from os import makedirs, path
# 导入 os 模块
import os

# 定义 mkdir_p 函数，用于创建目录，等效于命令行中的 mkdir -p
def mkdir_p(folder_path):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    """
    # 尝试创建目录
    try:
        # 使用 makedirs 函数创建目录
        makedirs(folder_path)
    # 如果抛出 OSError 异常
    except OSError as exc: # Python >2.5
        # 如果错误是因为文件已存在且目标路径确实是一个目录
        if exc.errno == EEXIST and path.isdir(folder_path):
            # 什么也不做，即忽略这个错误
            pass
        # 如果是其他类型的 OSError
        else:
            # 重新抛出这个异常
            raise

# 定义 searchForMaxIteration 函数，用于搜索文件夹中的最大迭代次数
def searchForMaxIteration(folder):
    # 通过列表推导式，从文件夹中的每个文件名提取出迭代次数并转换为整数
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    # 返回这些迭代次数中的最大值
    return max(saved_iters)
