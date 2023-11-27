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

# 导入argparse库中的ArgumentParser和Namespace，用于解析命令行参数。
from argparse import ArgumentParser, Namespace
import sys
import os

# 定义一个空类GroupParams，用作参数组的容器。
class GroupParams:
    pass

# 定义ParamGroup类，用于管理参数组。
class ParamGroup:
    # 初始化函数，接受一个解析器对象，组名，和一个布尔值fill_none。
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        # 创建一个参数组。
        group = parser.add_argument_group(name)
        # 遍历类中定义的所有变量。
        for key, value in vars(self).items():
            shorthand = False
            # 如果变量名以下划线开头，则为简写形式。
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            # 获取变量类型。
            t = type(value)
            # 如果fill_none为True，则将变量值设置为None。
            value = value if not fill_none else None 
            # 如果是简写形式的变量。
            if shorthand:
                # 如果变量类型为布尔型，则添加一个带有store_true行为的命令行参数。
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    # 否则，添加一个普通的命令行参数。
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                # 如果不是简写形式的变量。
                if t == bool:
                    # 如果变量类型为布尔型，则添加一个带有store_true行为的命令行参数。
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    # 否则，添加一个普通的命令行参数。
                    group.add_argument("--" + key, default=value, type=t)

    # 从命令行参数中提取与此参数组相关的参数。
    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            # 如果命令行参数中的变量在参数组中定义，则将其值设置到group对象上。
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

# 定义ModelParams类，继承自ParamGroup，用于管理模型加载相关的参数。
class ModelParams(ParamGroup): 
    # 初始化函数，接受一个解析器对象和一个布尔值sentinel，表示是否赋特值。
    def __init__(self, parser, sentinel=False):
        # 定义模型参数。
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        # 调用父类的初始化函数。
        super().__init__(parser, "Loading Parameters", sentinel)

    # 从命令行参数中提取模型参数，并进行处理。
    # 只将source_path更换为绝对路径
    def extract(self, args):
        g = super().extract(args)
        # 将source_path转换为绝对路径。
        g.source_path = os.path.abspath(g.source_path)
        return g

# 定义PipelineParams类，继承自ParamGroup，用于管理流水线相关的参数。
class PipelineParams(ParamGroup):
    # 初始化函数，接受一个解析器对象。
    def __init__(self, parser):
        # 设置一些流水线参数的默认值。
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        # 调用父类的初始化函数，并传递参数组的名称。
        super().__init__(parser, "Pipeline Parameters")

# 定义OptimizationParams类，继承自ParamGroup，用于管理优化过程的参数。
class OptimizationParams(ParamGroup):
    # 初始化函数，接受一个解析器对象。
    def __init__(self, parser):
        # 设置一系列优化参数的默认值。
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        # 调用父类的初始化函数，并传递参数组的名称。
        super().__init__(parser, "Optimization Parameters")

# 定义一个函数用于合并命令行参数和配置文件中的参数。
def get_combined_args(parser : ArgumentParser):
    # 获取命令行参数。
    cmdlne_string = sys.argv[1:]
    # 预设一个配置文件参数字符串。
    cfgfile_string = "Namespace()"
    # 解析命令行参数。
    args_cmdline = parser.parse_args(cmdlne_string)

    # 尝试从配置文件中读取参数。
    try:
        # 拼接配置文件路径。
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        # 打印查找配置文件的位置信息。
        print("Looking for config file in", cfgfilepath)
        # 打开配置文件。
        with open(cfgfilepath) as cfg_file:
            # 打印找到配置文件的信息。
            print("Config file found: {}".format(cfgfilepath))
            # 读取配置文件内容。
            cfgfile_string = cfg_file.read()
    except TypeError:
        # 如果遇到TypeError异常，打印配置文件未找到的信息。
        print("Config file not found at")
        pass
    # 评估配置文件字符串，将其转换为Namespace对象。
    args_cfgfile = eval(cfgfile_string)

    # 合并命令行参数和配置文件中的参数。
    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        # 如果命令行参数中的值不是None，则覆盖配置文件中的相应值。
        if v != None:
            merged_dict[k] = v
    # 返回合并后的参数集合。
    return Namespace(**merged_dict)
