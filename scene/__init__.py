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

# 导入 os 模块，用于执行与操作系统相关的操作，如文件和目录的管理
import os
# 导入 random 模块，用于生成随机数和执行与随机性相关的操作
import random
# 导入 json 模块，用于处理 JSON 数据格式，如读写 JSON 文件
import json
# 从 utils.system_utils 中导入 searchForMaxIteration 函数，用于搜索最大迭代次数
from utils.system_utils import searchForMaxIteration
# 从 scene.dataset_readers 中导入 sceneLoadTypeCallbacks，它包含不同类型场景的加载回调函数
from scene.dataset_readers import sceneLoadTypeCallbacks
# 从 scene.gaussian_model 中导入 GaussianModel 类，用于处理高斯模型相关的操作
from scene.gaussian_model import GaussianModel
# 从 arguments 模块中导入 ModelParams 类，用于处理模型参数
from arguments import ModelParams
# 从 utils.camera_utils 中导入 cameraList_from_camInfos 和 camera_to_JSON 函数，用于处理与相机信息相关的操作
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# 定义 Scene 类，用于表示和处理场景相关的数据和操作
class Scene:

    # 声明 gaussians 属性，类型为 GaussianModel，用于存储和处理高斯模型
    gaussians : GaussianModel

    # 初始化方法，用于创建 Scene 类的实例
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """

        # 将 args 中的 model_path 属性值赋给实例的 model_path 属性
        self.model_path = args.model_path

        # 初始化 loaded_iter 属性为 None，用于存储加载的迭代次数
        self.loaded_iter = None

        # 将传入的 gaussians 实例赋值给实例的 gaussians 属性
        self.gaussians = gaussians

        # 如果提供了 load_iteration 参数
        if load_iteration:
            # 如果 load_iteration 为 -1，表示寻找最大的迭代次数
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            # 否则，直接将提供的迭代次数赋值给 loaded_iter 属性
            else:
                self.loaded_iter = load_iteration
            # 打印加载训练模型的迭代次数信息
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化 train_cameras 字典，用于存储训练用的相机信息
        self.train_cameras = {}

        # 初始化 test_cameras 字典，用于存储测试用的相机信息
        self.test_cameras = {}

        # 检查 args.source_path 路径下是否存在 "sparse" 文件夹
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # 使用 Colmap 回调函数加载场景信息
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        # 检查 args.source_path 路径下是否存在 "transforms_train.json" 文件
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # 打印发现特定文件的信息，并假定为 Blender 数据集
            print("Found transforms_train.json file, assuming Blender data set!")
            # 使用 Blender 回调函数加载场景信息
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # 如果以上条件都不满足，则无法识别场景类型
        else:
            assert False, "Could not recognize scene type!"

        # 如果没有指定加载的迭代次数
        if not self.loaded_iter:
            # 以二进制读模式打开源 PLY 文件，并以二进制写模式打开目标文件进行写入
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                # 将源文件的内容写入目标文件
                dest_file.write(src_file.read())

            # 初始化用于存储相机信息的 JSON 对象列表
            json_cams = []
            # 初始化一个列表来合并测试和训练相机信息
            camlist = []

            # 如果场景信息中包含测试用相机
            if scene_info.test_cameras:
                # 将测试相机添加到相机列表中
                camlist.extend(scene_info.test_cameras)

            # 如果场景信息中包含训练用相机
            if scene_info.train_cameras:
                # 将训练相机也添加到相机列表中
                camlist.extend(scene_info.train_cameras)

            # 遍历合并后的相机列表
            for id, cam in enumerate(camlist):
                # 将每个相机的信息转换为 JSON 格式并添加到 json_cams 列表中
                json_cams.append(camera_to_JSON(id, cam))

            # 将相机信息的 JSON 列表写入文件
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                # 使用 json.dump 将 JSON 对象写入文件
                json.dump(json_cams, file)

        # 如果 shuffle 参数为 True，即需要对相机信息进行随机排序
        if shuffle:
            # 对训练相机列表进行随机排序
            random.shuffle(scene_info.train_cameras)  # 多分辨率一致的随机排序
            # 对测试相机列表也进行随机排序
            random.shuffle(scene_info.test_cameras)  # 多分辨率一致的随机排序

        # 设置相机范围为场景信息中的 nerf 正规化半径
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 遍历分辨率比例列表
        for resolution_scale in resolution_scales:
            # 打印正在加载训练相机的信息
            print("Loading Training Cameras")
            # 根据分辨率比例和场景信息加载训练相机列表
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # 打印正在加载测试相机的信息
            print("Loading Test Cameras")
            # 根据分辨率比例和场景信息加载测试相机列表
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 如果指定了加载的迭代次数
        if self.loaded_iter:
            # 加载指定迭代次数的 PLY 文件
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                "point_cloud",
                                                "iteration_" + str(self.loaded_iter),
                                                "point_cloud.ply"))
        # 如果没有指定加载的迭代次数
        else:
            # 从点云数据创建高斯模型
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # 定义 save 方法，用于保存场景数据
        def save(self, iteration):
            # 构建点云路径，包含指定的迭代次数
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            # 保存高斯模型到指定路径的 PLY 文件
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        # 定义 getTrainCameras 方法，用于获取训练相机
        def getTrainCameras(self, scale=1.0):
            # 返回指定分辨率比例下的训练相机列表
            return self.train_cameras[scale]

        # 定义 getTestCameras 方法，用于获取测试相机
        def getTestCameras(self, scale=1.0):
            # 返回指定分辨率比例下的测试相机列表
            return self.test_cameras[scale]