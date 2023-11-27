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

# 导入os模块，用于与操作系统进行交互，如文件和目录操作
import os

# 导入PyTorch库，用于深度学习模型的创建和训练
import torch

# 从random模块导入randint函数，用于生成随机整数
from random import randint

# 从utils.loss_utils模块导入l1_loss和ssim函数，用于计算损失
from utils.loss_utils import l1_loss, ssim

# 从gaussian_renderer模块导入render函数和network_gui对象，用于渲染和网络界面
from gaussian_renderer import render, network_gui

# 导入sys模块，用于访问与Python解释器相关的变量和功能
import sys

# 从scene模块导入Scene和GaussianModel类，用于场景表示和高斯模型处理
from scene import Scene, GaussianModel

# 从utils.general_utils模块导入safe_state函数，用于设置安全状态
from utils.general_utils import safe_state

# 导入uuid模块，用于生成唯一标识符
import uuid

# 从tqdm模块导入tqdm类，用于显示进度条
from tqdm import tqdm

# 从utils.image_utils模块导入psnr函数，用于计算峰值信噪比
from utils.image_utils import psnr

# 从argparse模块导入ArgumentParser和Namespace，用于解析命令行参数
from argparse import ArgumentParser, Namespace

# 从arguments模块导入ModelParams, PipelineParams, OptimizationParams类，用于参数配置
from arguments import ModelParams, PipelineParams, OptimizationParams

# 尝试导入torch.utils.tensorboard模块中的SummaryWriter类，用于记录日志；如果失败，设置TENSORBOARD_FOUND标志为False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 定义一个用于训练的函数，接受数据集、优化选项、管道对象、测试、保存和检查点的迭代次数等参数
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    
    # 初始化迭代次数
    first_iter = 0

    # 准备输出和日志记录，返回一个TensorBoard写入器
    tb_writer = prepare_output_and_logger(dataset)

    # 创建一个GaussianModel实例，用于渲染和处理高斯模型
    gaussians = GaussianModel(dataset.sh_degree)

    # 创建一个Scene实例，用于场景管理
    scene = Scene(dataset, gaussians)

    # 对高斯模型进行训练设置
    gaussians.training_setup(opt)

    # 如果提供了检查点，从中加载模型参数和起始迭代次数，并恢复高斯模型状态
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 根据数据集是否有白色背景设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # 将背景颜色转换为PyTorch张量，并放到CUDA设备上（GPU）
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建用于测量迭代时间的CUDA事件对象
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 初始化视点堆栈和用于日志记录的EMA损失
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    # 初始化一个进度条，用于展示训练进度
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # 开始迭代训练过程
    for iteration in range(first_iter, opt.iterations + 1):        
        # 检查网络GUI的连接状态，尝试建立连接
        if network_gui.conn == None:
            network_gui.try_connect()

        # 当网络GUI连接存在时，进行数据接收和发送处理
        while network_gui.conn != None:
            try:
                # 初始化网络图像字节为None
                net_image_bytes = None
                # 从网络GUI接收数据
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()

                # 如果收到自定义相机设置，进行渲染并转换图像为字节
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                # 将图像字节数据发送回网络GUI
                network_gui.send(net_image_bytes, dataset.source_path)

                # 检查是否需要进行训练或终止训练
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                # 如有异常，断开网络GUI连接
                network_gui.conn = None

        # 记录迭代开始时间
        iter_start.record()

        # 更新高斯模型的学习率
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代，增加球谐（SH）的级别，直到达到最大程度
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 如果视点堆栈为空，则从场景中获取训练用的相机视点并复制
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # 从视点堆栈中随机选择一个相机视点
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 如果迭代数等于调试开始点，则开启调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # 使用选择的相机视点进行渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # 从渲染结果中提取图像及其他相关数据
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 计算损失
        # 获取当前视点的原始图像，并转移到CUDA设备
        gt_image = viewpoint_cam.original_image.cuda()
        # 计算L1损失
        Ll1 = l1_loss(image, gt_image)
        # 计算总损失，结合L1损失和结构相似性指数（SSIM）损失
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # 反向传播损失
        loss.backward()

        # 记录迭代结束时间
        iter_end.record()

        # 在不计算梯度的情况下执行以下操作
        with torch.no_grad():
            # 更新进度条的损失显示
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # 每10次迭代更新一次进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            # 如果达到最后一次迭代，关闭进度条
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录训练过程并保存相关信息
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # 如果当前迭代数在保存迭代列表中，打印保存信息并保存当前场景状态
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密集化（Densification）过程
            if iteration < opt.densify_until_iter:
                # 跟踪图像空间中最大半径，用于后续的修剪
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 向高斯模型添加密集化统计数据
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 如果达到开始密集化的迭代次数，并且处于密集化间隔，执行密集化和修剪
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 设置大小阈值，用于确定何时重置不透明度
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 执行密集化和修剪操作
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # 如果迭代数是不透明度重置间隔的倍数，或在特定条件下，重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步骤
            # 如果迭代数小于总迭代次数，执行优化器步骤
            if iteration < opt.iterations:
                # 执行优化步骤
                gaussians.optimizer.step()
                # 重置优化器梯度
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 检查点保存
            # 如果当前迭代次数在检查点迭代列表中，保存当前模型状态为检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # 保存高斯模型状态和迭代次数
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

# 定义一个函数，用于准备输出目录和日志记录器
def prepare_output_and_logger(args):    
    # 检查是否提供了模型路径，如果没有，则生成一个唯一的路径
    if not args.model_path:
        # 尝试获取环境变量'OAR_JOB_ID'，如果不存在则使用uuid生成一个唯一字符串
        # OAR在一般的个人电脑上都是没有的，所以会直接用uuid生成模型的唯一路径名称
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        # 设置模型路径为'./output/'目录下的唯一字符串的前10个字符
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 设置输出文件夹，并打印输出文件夹路径
    print("Output folder: {}".format(args.model_path))
    # 创建输出文件夹，如果已存在则不会报错
    os.makedirs(args.model_path, exist_ok = True)
    # 在输出文件夹中创建一个配置参数文件，并将参数写入该文件
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 初始化TensorBoard日志记录器
    tb_writer = None
    # 如果检测到TensorBoard模块，则创建一个TensorBoard日志记录器；否则，打印不可用信息
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    # 返回TensorBoard日志记录器对象
    return tb_writer

# 定义一个函数，用于生成训练报告，记录训练过程中的各种指标
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    # 如果提供了TensorBoard写入器，记录L1损失、总损失和迭代时间
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 如果当前迭代数在测试迭代列表中，进行测试和训练集样本的报告
    if iteration in testing_iterations:
        # 清空CUDA缓存以释放未使用的内存
        torch.cuda.empty_cache()
        # 设置验证配置，包括测试和训练数据的相机视点
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # 遍历测试和训练配置
        for config in validation_configs:
            # 如果相机列表不为空
            if config['cameras'] and len(config['cameras']) > 0:
                # 初始化L1损失和PSNR测试变量
                l1_test = 0.0
                psnr_test = 0.0
                # 对每个相机视点进行迭代
                for idx, viewpoint in enumerate(config['cameras']):
                    # 渲染图像并将其裁剪到0和1之间
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    # 获取原始图像并将其转移到CUDA设备
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # 如果提供了TensorBoard写入器，记录渲染和真实图像
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # 累加L1损失和PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                # 计算平均L1损失和PSNR
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # 打印评估结果
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # 如果提供了TensorBoard写入器，记录L1损失和PSNR
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 如果提供了TensorBoard写入器，记录不透明度直方图和高斯点总数
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        # 再次清空CUDA缓存
        torch.cuda.empty_cache()

# 检查是否直接运行这个脚本，而不是导入
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")

    # 初始化模型参数-即dataset
    # sh_degree: Spherical Harmonics的度数，设置为3。
    # _source_path: 模型源文件的路径，初始为空字符串。
    # _model_path: 模型文件的路径，初始为空字符串。
    # _images: 用于存储模型图像的文件夹，默认为"images"。
    # _resolution: 图像分辨率，默认为-1，表示不指定。
    # _white_background: 是否使用白色背景，默认为False。
    # data_device: 数据处理设备，设置为'cuda'表示使用CUDA。
    # eval: 是否处于评估模式，默认为False。
    lp = ModelParams(parser)

    # 初始化优化参数
    # iterations: 优化过程中的迭代次数，默认为30,000次。
    # position_lr_init: 位置学习率的初始值，默认为0.00016。
    # position_lr_final: 位置学习率的最终值，默认为0.0000016。
    # position_lr_delay_mult: 位置学习率延迟乘数，默认为0.01。
    # position_lr_max_steps: 位置学习率的最大步数，默认为30,000。
    # feature_lr: 特征学习率，默认为0.0025。
    # opacity_lr: 透明度学习率，默认为0.05。
    # scaling_lr: 缩放学习率，默认为0.005。
    # rotation_lr: 旋转学习率，默认为0.001。
    # percent_dense: 密集百分比，用于控制优化过程中的密集度，默认为0.01（1%）。
    # lambda_dssim: DSSIM（结构相似性）损失函数的权重，默认为0.2。
    # densification_interval: 密集化间隔，指定多少次迭代进行一次密集化，默认为每100次迭代。
    # opacity_reset_interval: 透明度重置间隔，指定多少次迭代后重置透明度，默认为每3000次迭代。
    # densify_from_iter: 从第几次迭代开始进行密集化，默认从第500次迭代开始。
    # densify_until_iter: 进行密集化直到第几次迭代，此参数设置为15,000，意味着直到第15,000次迭代之前都会进行密集化。
    # densify_grad_threshold: 密集化梯度阈值，用于控制何时进行密集化，默认值为0.0002。
    op = OptimizationParams(parser)

    # 初始化管道参数
    # convert_SHs_python: 布尔值，指示是否使用Python来转换Spherical Harmonics（SHs）。默认为False，即不使用Python进行SHs转换。
    # compute_cov3D_python: 布尔值，指示是否使用Python来计算3D协方差。默认为False，即不使用Python进行3D协方差计算。
    # debug: 布尔值，用于启用或禁用调试模式。默认为False，即在默认情况下不启用调试模式。
    pp = PipelineParams(parser)

    # 添加IP地址参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    # 添加端口号参数
    parser.add_argument('--port', type=int, default=6009)
    # 添加调试开始的迭代次数参数
    parser.add_argument('--debug_from', type=int, default=-1)
    # 添加异常检测开关参数
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # 添加测试迭代次数参数
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # 添加保存迭代次数参数
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # 添加安静模式开关参数
    parser.add_argument("--quiet", action="store_true")
    # 添加检查点迭代次数参数
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # 添加起始检查点参数
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # 解析命令行输入的参数
    args = parser.parse_args(sys.argv[1:])
    # 将总迭代次数添加到保存迭代列表中
    args.save_iterations.append(args.iterations)
    
    # 输出当前优化的模型路径
    print("Optimizing " + args.model_path)

    # 初始化系统状态（随机数生成器）
    safe_state(args.quiet)

    # 默认打开 http://127.0.0.1:6009 可视化训练流程
    # 初始化GUI服务器，配置并运行训练
    network_gui.init(args.ip, args.port)
    # 设置是否检测自动梯度异常
    # PyTorch 会在自动微分过程中进行额外的检查，以捕捉可能导致计算错误的操作。
    # 这种检测对于调试非常有用，因为它可以帮助识别诸如 NaN（不是数字）或 Inf（无穷大）值的出现原因。
    # 这些值通常是由于数值不稳定的操作（如除以零）或其他错误造成的。
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # 执行训练函数
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # 训练完成后打印信息
    print("\nTraining complete.")
