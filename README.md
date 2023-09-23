# 用于实时辐射场渲染的3D高斯喷射

Bernhard Kerbl*，Georgios Kopanas*，Thomas Leimkühler，George Drettakis（*表示平等贡献）


| [网页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [完整论文](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) | [视频](https://youtu.be/T_kXY43VZnk) | [其他GRAPHDECO出版物](http://www-sop.inria.fr/reves/publis/gdindex.php) | [FUNGRAPH项目页面](https://fungraph.inria.fr) |

| [T&T+DB COLMAP（650MB）](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) | [预训练模型（14 GB）](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) | [Windows查看器（60MB）](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip) | [评估图像（7 GB）](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip) |


![预览图像](assets/teaser.png)

此存储库包含与论文“用于实时辐射场渲染的3D高斯喷射”相关的官方作者实现，该论文可在[此处](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)找到。我们还提供用于创建论文中报告的错误指标的参考图像，以及最近创建的预训练模型。


摘要：辐射场方法最近彻底改变了用多张照片或视频捕捉的场景的新视图合成。然而，要实现高视觉质量仍然需要训练和渲染成本高昂的神经网络，而最近的更快方法不可避免地以速度为代价牺牲质量。对于无界和完整的场景（而不是孤立的对象）以及1080p分辨率的渲染，目前没有方法能够实现实时显示速率。我们引入了三个关键元素，使我们能够在保持有竞争力的训练时间的同时实现最先进的视觉质量，并且重要的是允许高质量的实时（≥ 30 fps）新视图合成在1080p分辨率。首先，从相机校准过程中产生的稀疏点开始，我们用3D高斯表示场景，这些高斯保留了连续体积辐射场在场景优化中的理想属性，同时避免了在空白空间中进行不必要的计算；其次，我们执行3D高斯的交错优化/密度控制，特别是优化各向异性协方差以实现场景的准确表示；第三，我们开发了一种快速的可见性感知渲染算法，该算法支持各向异性喷射，并加速了训练和实时渲染。我们在几个既定的数据集上展示了最先进的视觉质量和实时渲染。


## 逐步教程

Jonathan Stephens制作了一个出色的逐步教程，用于在您的机器上设置高斯喷射，以及用于从视频创建可用数据集的说明。如果下面的说明对您来说太枯燥了，请继续并在[这里](https://www.youtube.com/watch?v=UXtuigy_wYc)查看。

## Colab

用户[camenduru](https://github.com/camenduru)非常友好地提供了一个Colab模板，该模板使用此存储库的源代码（状态：2023年8月！）以便快速轻松地访问该方法。请在[这里](https://github.com/camenduru/gaussian-splatting-colab)查看。

## 克隆存储库

该存储库包含子模块，因此请使用以下命令进行检出：
```shell
# SSH
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
```
或
```shell
# HTTPS
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```


## 概览

代码库主要包含四个组件：
- 一个基于PyTorch的优化器，用于从SfM（结构从运动）输入中生成3D高斯模型
- 一个网络查看器，允许连接并可视化优化过程
- 一个基于OpenGL的实时查看器，用于实时渲染训练过的模型
- 一个脚本，用于帮助您将自己的图像转换为可用于优化的SfM数据集

这些组件在硬件和软件方面有不同的需求。它们已在Windows 10和Ubuntu Linux 22.04上进行了测试。下面各节中有设置和运行它们的说明。

## 优化器

优化器在Python环境中使用PyTorch和CUDA扩展来生成训练过的模型。

### 硬件需求

- 支持CUDA的GPU，计算能力需为7.0+
- 24 GB显存（以达到论文评估质量）
- 有关较小显存配置，请参见常见问题解答

### 软件需求
- Conda（推荐用于简单设置）
- 用于PyTorch扩展的C++编译器（我们在Windows上使用了Visual Studio 2019）
- 用于PyTorch扩展的CUDA SDK 11，*在* Visual Studio之后安装（我们使用了11.8，**11.6存在已知问题**）
- C++编译器和CUDA SDK必须是兼容的



### 设置

#### 本地设置

我们提供的默认安装方法是基于Conda包和环境管理的：
```shell
SET DISTUTILS_USE_SDK=1 # 仅限Windows
conda env create --file environment.yml
conda activate gaussian_splatting
```
请注意，此过程假定您已安装了CUDA SDK **11**，而不是**12**。有关修改，请参见下文。

提示：使用Conda下载包并创建新环境可能需要大量的磁盘空间。默认情况下，Conda将使用主系统硬盘。您可以通过指定不同的包下载位置和在不同驱动器上的环境来避免这一点：

```shell
conda config --add pkgs_dirs <驱动器>/<包路径>
conda env create --file environment.yml --prefix <驱动器>/<环境路径>/gaussian_splatting
conda activate <驱动器>/<环境路径>/gaussian_splatting
```

#### 修改

如果您的磁盘空间充足，我们建议使用我们的环境文件来设置与我们相同的训练环境。如果您想进行修改，请注意主版本更改可能会影响我们方法的结果。然而，我们的（有限的）实验表明，代码库在更为更新的环境中（Python 3.8，PyTorch 2.0.0，CUDA 12）运行得相当不错。请确保创建一个环境，其中PyTorch和其CUDA运行时版本匹配，并且已安装的CUDA SDK与PyTorch的CUDA版本没有主版本差异。

#### 已知问题

一些用户在Windows上构建子模块时遇到问题（```cl.exe: 文件未找到```或类似）。请参考常见问题解答中针对此问题的解决方法。


### 运行

要运行优化器，只需使用以下命令：

```shell
python train.py -s <指向COLMAP或NeRF Synthetic数据集的路径>
```

<details>
<summary><span style="font-weight: bold;">train.py的命令行参数</span></summary>

  #### --source_path / -s
  指向包含COLMAP或Synthetic NeRF数据集的源目录的路径。
  #### --model_path / -m 
  存储训练模型的路径（默认为```output/<随机>```）。
  #### --images / -i
  COLMAP图像的替代子目录（默认为```images```）。
  #### --eval
  添加此标志以使用MipNeRF360风格的训练/测试分割进行评估。
  #### --resolution / -r
  指定训练前加载图像的分辨率。如果提供了```1, 2, 4```或```8```，则分别使用原始、1/2、1/4或1/8的分辨率。对于所有其他值，保持图像的纵横比，将宽度调整为给定的数字。**如果未设置且输入图像宽度超过1.6K像素，输入将自动缩放到此目标。**

  #### --data_device
  指定将源图像数据放在哪里，默认为```cuda```，如果在大型/高分辨率数据集上进行训练，建议使用```cpu```，这将减少VRAM消耗，但会稍微减慢训练速度。感谢[HrsPythonix](https://github.com/HrsPythonix)。

  #### --white_background / -w
  添加此标志以使用白色背景而不是黑色（默认），例如，用于评估NeRF Synthetic数据集。

  #### --sh_degree
  要使用的球面谐波的阶数（不大于3）。默认为```3```。

  #### --convert_SHs_python
  标志，使管道用PyTorch而不是我们的方法计算SHs（球面谐波）的正向和反向。

  #### --convert_cov3D_python
  标志，使管道用PyTorch而不是我们的方法计算3D协方差的正向和反向。

  #### --debug
  如果您遇到错误，启用调试模式。如果光栅器失败，将创建一个```dump```文件，您可以将其转发给我们，以便我们进行检查。

  #### --debug_from
  调试是**缓慢的**。您可以指定一个迭代（从0开始），在该迭代之后上述调试变为活动状态。

  #### --iterations
  要训练的总迭代次数，默认为```30_000```。

  #### --ip
  启动GUI服务器的IP，默认为```127.0.0.1```。

  #### --port
  用于GUI服务器的端口，默认为```6009```。

  #### --test_iterations
  以空格分隔的迭代次数，在这些迭代次数上，训练脚本会计算L1和PSNR在测试集上的值，默认为```7000 30000```。

  #### --save_iterations
  以空格分隔的迭代次数，在这些迭代次数上，训练脚本会保存高斯模型，默认为```7000 30000 <iterations>```。

  #### --checkpoint_iterations
  以空格分隔的迭代次数，在这些迭代次数上存储一个检查点以便以后继续，保存在模型目录中。

  #### --start_checkpoint
  要从中继续训练的已保存检查点的路径。

  #### --quiet
  标志，用于省略写入标准输出管道的任何文本。

  #### --feature_lr
  球面谐波特征的学习率，默认为```0.0025```。

  #### --opacity_lr
  不透明度的学习率，默认为```0.05```。

  #### --scaling_lr
  缩放的学习率，默认为```0.005```。

  #### --rotation_lr
  旋转的学习率，默认为```0.001```。

  #### --position_lr_max_steps
  位置学习率从```initial```到```final```的步数（从0开始）。默认为```30_000```。

  #### --position_lr_init
  初始3D位置的学习率，默认为```0.00016```。

  #### --position_lr_final
  最终3D位置的学习率，默认为```0.0000016```。

  #### --position_lr_delay_mult
  位置学习率的乘数（参见Plenoxels），默认为```0.01```。

  #### --densify_from_iter
  开始密集化的迭代，默认为```500```。

  #### --densify_until_iter
  停止密集化的迭代，默认为```15_000```。

  #### --densify_grad_threshold
  基于2D位置梯度决定是否应该密集化点的限制，默认为```0.0002```。

  #### --densification_interal
  多久进行一次密集化，默认为```100```（每100次迭代）。

  #### --opacity_reset_interval
  多久重置一次不透明度，默认为```3_000```。

  #### --lambda_dssim
  SSIM对总损失的影响，从0到1，默认为```0.2```。

  #### --percent_dense
  场景范围（0--1）的百分比，点必须超过该百分比才能被强制密集化，默认为```0.01```。

</details>
<br>

请注意，与MipNeRF360类似，我们针对的图像分辨率在1-1.6K像素范围内。为了方便起见，可以传递任意大小的输入，如果其宽度超过1600像素，将自动调整大小。我们建议保持这种行为，但您也可以通过设置```-r 1```来强制训练使用您的高分辨率图像。

MipNeRF360的场景由论文作者在[这里](https://jonbarron.info/mipnerf360/)托管。您可以在[这里](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)找到我们的Tanks&Temples和Deep Blending的SfM数据集。如果您没有提供输出模型目录（```-m```），训练后的模型将写入```output```目录中随机唯一名称的文件夹。此时，可以使用实时查看器（见下文）查看训练后的模型。

### 评估
默认情况下，训练模型使用数据集中的所有可用图像。要在保留测试集进行评估的同时训练它们，请使用```--eval```标志。这样，您可以按如下方式渲染训练/测试集并生成错误指标：
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # 使用训练/测试集进行训练
python render.py -m <path to trained model> # 生成渲染图像
python metrics.py -m <path to trained model> # 计算渲染图像上的错误指标
```

如果您想评估我们的[预训练模型](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)，您将需要下载相应的源数据集，并通过额外的```--source_path/-s```标志指示其位置给```render.py```。注意：预训练模型是使用发布代码库创建的。该代码库已经过清理并包括错误修复，因此您从中获得的评估指标将与论文中的不同。
```shell
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```


<details>
<summary><span style="font-weight: bold;">render.py的命令行参数</span></summary>

  #### --model_path / -m 
  您想为其创建渲染的训练模型目录的路径。
  #### --skip_train
  标志以跳过渲染训练集。
  #### --skip_test
  标志以跳过渲染测试集。
  #### --quiet 
  标志以省略写入标准输出管道的任何文本。

  **以下参数将根据用于训练的内容自动从模型路径中读取。然而，您可以通过在命令行上明确提供它们来覆盖它们。**

  #### --source_path / -s
  包含COLMAP或Synthetic NeRF数据集的源目录的路径。
  #### --images / -i
  COLMAP图像的替代子目录（默认为```images```）。
  #### --eval
  添加此标志以使用MipNeRF360风格的训练/测试分割进行评估。
  #### --resolution / -r
  在训练之前更改加载图像的分辨率。如果提供```1, 2, 4```或```8```，则分别使用原始、1/2、1/4或1/8的分辨率。对于所有其他值，保持图像纵横比的同时，将宽度重新缩放为给定的数字。默认为```1```。
  #### --white_background / -w
  添加此标志以使用白色背景而不是黑色（默认），例如，用于评估NeRF Synthetic数据集。
  #### --convert_SHs_python
  标志以使管道使用PyTorch而不是我们的计算的SHs进行渲染。
  #### --convert_cov3D_python
  标志以使管道使用PyTorch而不是我们的计算的3D协方差进行渲染。

</details>


<details>
<summary><span style="font-weight: bold;">metrics.py的命令行参数</span></summary>

  #### --model_paths / -m 
  应计算指标的模型路径的空格分隔列表。
</details>
<br>

我们还提供了```full_eval.py```脚本。该脚本指定了我们评估中使用的例程，并演示了一些附加参数的使用，例如，```--images (-i)```用于在COLMAP数据集中定义替代图像目录。如果您已下载并提取了所有训练数据，您可以像这样运行它：
```shell
python full_eval.py -m360 <mipnerf360文件夹> -tat <坦克和庙宇文件夹> -db <深度混合文件夹>
```
在当前版本中，该过程在包含A6000的参考机器上大约需要7小时。如果您想对我们的预训练模型进行完整评估，您可以指定它们的下载位置并跳过训练。
```shell
python full_eval.py -o <预训练模型的目录> --skip_training -m360 <mipnerf360文件夹> -tat <坦克和庙宇文件夹> -db <深度混合文件夹>
```

如果您想根据我们论文中的[评估图像](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip)计算指标，您也可以跳过渲染。在这种情况下，不需要提供源数据集。您可以一次计算多个图像集的指标。
```shell
python full_eval.py -m <评估图像的目录>/garden ... --skip_training --skip_rendering
```


<details>
<summary><span style="font-weight: bold;">full_eval.py的命令行参数</span></summary>

  #### --skip_training
  标志用于跳过训练阶段。
  #### --skip_rendering
  标志用于跳过渲染阶段。
  #### --skip_metrics
  标志用于跳过指标计算阶段。
  #### --output_path
  用于存放渲染和结果的目录，默认为```./eval```，如果评估预训练模型，则设置为预训练模型的位置。
  #### --mipnerf360 / -m360
  MipNeRF360源数据集的路径，如果进行训练或渲染则需要。
  #### --tanksandtemples / -tat
  Tanks&Temples源数据集的路径，如果进行训练或渲染则需要。
  #### --deepblending / -db
  Deep Blending源数据集的路径，如果进行训练或渲染则需要。
</details>
<br>


## 交互式查看器
我们为我们的方法提供了两种交互式查看器：远程和实时。我们的查看解决方案基于[SIBR](https://sibr.gitlabpages.inria.fr/)框架，该框架由GRAPHDECO团队为多个新视图合成项目开发。

### 硬件要求
- 支持OpenGL 4.5的GPU和驱动程序（或最新的MESA软件）
- 建议使用4 GB VRAM
- 支持计算能力7.0+的CUDA准备就绪的GPU（仅用于实时查看器）

### 软件要求
- Visual Studio或g++，**不是Clang**（我们在Windows上使用了Visual Studio 2019）
- 在Visual Studio之后安装CUDA SDK 11（我们使用了11.8）
- CMake（最近版本，我们使用了3.24）
- 7zip（仅在Windows上）

### 预构建的Windows二进制文件
我们在[这里](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip)为Windows提供了预构建的二进制文件。我们建议在Windows上使用它们进行高效的设置，因为SIBR的构建涉及到必须即时下载和编译的多个外部依赖项。

### 从源代码安装
如果您使用子模块克隆（例如，使用```--recursive```），则查看器的源代码位于```SIBR_viewers```中。网络查看器在用于基于图像的渲染应用程序的SIBR框架内运行。

#### Windows
CMake应该会处理您的依赖项。
```shell
cd SIBR_viewers
cmake -Bbuild .
cmake --build build --target install --config RelWithDebInfo
```
您可以指定不同的配置，例如，如果您在开发过程中需要更多的控制，可以使用```Debug```。


#### Ubuntu 22.04
在运行项目设置之前，您需要安装一些依赖项。
```shell
# 依赖项
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# 项目设置
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # 添加 -G Ninja 以更快地构建
cmake --build build -j24 --target install
```

#### Ubuntu 20.04
与Focal Fossa的向后兼容性没有完全测试，但在执行以下命令后，使用CMake构建SIBR仍应该可行。
```shell
git checkout fossa_compatibility
```

### SIBR查看器中的导航
SIBR界面提供了多种导航场景的方法。默认情况下，您将以FPS导航器开始，您可以使用```W, A, S, D, Q, E```进行相机平移，使用```I, K, J, L, U, O```进行旋转。或者，您可能希望使用Trackball风格的导航器（从浮动菜单中选择）。您还可以使用```Snap to```按钮跳转到数据集中的相机，或使用```Snap to closest```找到最近的相机。浮动菜单还允许您更改导航速度。您可以使用```Scaling Modifier```来控制显示的高斯函数的大小，或显示初始点云。



### 运行网络查看器

![GitHub资源](https://github.com/graphdeco-inria/gaussian-splatting/assets/40643808/90a2e4d3-cf2e-4633-b35f-bfe284e28ff7)

在提取或安装查看器之后，您可以在```<SIBR安装目录>/bin```中运行已编译的```SIBR_remoteGaussian_app[_config]```应用程序，例如：
```shell
./<SIBR安装目录>/bin/SIBR_remoteGaussian_app
```
网络查看器允许您连接到在同一台或不同机器上运行的训练过程。如果您在同一台机器和操作系统上进行训练，则不应需要命令行参数：优化器会将训练数据的位置通知给网络查看器。默认情况下，优化器和网络查看器将尝试在**localhost**上的**6009**端口建立连接。您可以通过为优化器和网络查看器提供匹配的```--ip```和```--port```参数来更改此行为。如果由于某种原因优化器用于查找训练数据的路径对网络查看器不可达（例如，由于它们在不同的（虚拟）机器上运行），您可以使用```-s <源路径>```指定一个覆盖位置给查看器。

<details>
<summary><span style="font-weight: bold;">网络查看器的主要命令行参数</span></summary>

  #### --path / -s
  参数，用于覆盖模型的源数据集路径。
  #### --ip
  用于连接到正在运行的训练脚本的IP。
  #### --port
  用于连接到正在运行的训练脚本的端口。
  #### --rendering-size
  接受两个由空格分隔的数字，以定义网络渲染发生的分辨率，默认宽度为```1200```。
  请注意，要强制与输入图像不同的纵横比，您还需要```--force-aspect-ratio```。
  #### --load_images
  标志，用于加载源数据集图像，以便在每个相机的顶视图中显示。
</details>
<br>



### 运行实时查看器

![GitHub资源](https://github.com/graphdeco-inria/gaussian-splatting/assets/40643808/0940547f-1d82-4c2f-a616-44eabbf0f816)

在提取或安装查看器之后，您可以在```<SIBR安装目录>/bin```中运行已编译的```SIBR_gaussianViewer_app[_config]```应用程序，例如：
```shell
./<SIBR安装目录>/bin/SIBR_gaussianViewer_app -m <训练模型的路径>
```

提供指向训练模型目录的```-m```参数应该就足够了。或者，您可以使用```-s```指定训练输入数据的覆盖位置。要使用除自动选择之外的特定分辨率，请指定```--rendering-size <宽度> <高度>```。如果您希望确切的分辨率并且不介意图像失真，可以与```--force-aspect-ratio```结合使用。

**要解锁完整的帧率，请在您的机器上以及应用程序中（菜单 → 显示）禁用V-Sync。在多GPU系统（例如，笔记本电脑）中，您的OpenGL/显示GPU应与您的CUDA GPU相同（例如，通过在Windows上设置应用程序的GPU首选项）以获得最大性能。**

![示例图片](assets/select.png)

除了初始点云和斑点之外，您还可以选择通过从浮动菜单中渲染它们作为椭球体来可视化高斯函数。SIBR具有许多其他功能，请参阅[文档](https://sibr.gitlabpages.inria.fr/)以获取有关查看器、导航选项等的更多详细信息。还有一个顶视图（从菜单中可用），显示输入相机的位置和原始的SfM点云；请注意，启用顶视图时会减慢渲染速度。实时查看器还使用稍微更激进、更快的剔除，这可以在浮动菜单中切换。如果您遇到可以通过关闭快速剔除来解决的问题，请告诉我们。


<details>
<summary><span style="font-weight: bold;">实时查看器的主要命令行参数</span></summary>

  #### --model-path / -m
  指向训练模型的路径。
  #### --iteration
  如果有多个状态可用，指定要加载哪个状态。默认为最新可用的迭代。
  #### --path / -s
  参数用于覆盖模型对源数据集的路径。
  #### --rendering-size
  接受两个由空格分隔的数字，以定义实时渲染发生的分辨率，默认宽度为```1200```。注意，要强制与输入图像不同的纵横比，您需要```--force-aspect-ratio```。
  #### --load_images
  标志用于加载源数据集图像，以便在每个相机的顶视图中显示。
  #### --device
  如果有多个CUDA设备可用，用于光栅化的CUDA设备的索引，默认为```0```。
  #### --no_interop
  强制禁用CUDA/GL互操作。在可能不按规范行事的系统上使用（例如，具有MESA GL 4.5软件渲染的WSL2）。
</details>
<br>


## 处理您自己的场景

我们的COLMAP加载器期望在源路径位置找到以下数据集结构：

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

对于光栅化，相机模型必须是SIMPLE_PINHOLE或PINHOLE相机。我们提供了一个转换脚本 ```convert.py```，用于从输入图像中提取未失真的图像和SfM信息。您还可以选择使用ImageMagick来调整未失真图像的大小。这种重新缩放与MipNeRF360类似，即在相应的文件夹中创建具有原始分辨率的1/2、1/4和1/8的图像。要使用它们，请首先安装COLMAP（理想情况下是支持CUDA的）和ImageMagick的最新版本。将您想要使用的图像放在目录 ```<location>/input``` 中。

```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```

如果您的系统路径上有COLMAP和ImageMagick，您可以简单地运行：

```shell
python convert.py -s <location> [--resize] #如果不调整大小，不需要ImageMagick
```

或者，您可以使用可选参数 ```--colmap_executable``` 和 ```--magick_executable``` 指向各自的路径。请注意，在Windows上，可执行文件应指向负责设置执行环境的COLMAP ```.bat``` 文件。完成后，```<location>``` 将包含预期的COLMAP数据集结构，其中包括未失真、调整大小的输入图像，以及您的原始图像和目录 ```distorted``` 中的一些临时（失真）数据。

如果您有自己的COLMAP数据集但没有失真（例如，使用 ```OPENCV``` 相机），您可以尝试仅运行脚本的最后一部分：将图像放在 ```input``` 中，并将COLMAP信息放在一个名为 ```distorted``` 的子目录中：

```
<location>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---distorted
    |---database.db
    |---sparse
        |---0
            |---...
```

然后运行：

```shell
python convert.py -s <location> --skip_matching [--resize] #如果不调整大小，不需要ImageMagick
```


<details>
<summary><span style="font-weight: bold;">convert.py 的命令行参数</span></summary>

  #### --no_gpu
  标志，用于避免在COLMAP中使用GPU。
  #### --skip_matching
  标志，用于指示图像的COLMAP信息是可用的。
  #### --source_path / -s
  输入的位置。
  #### --camera 
  在早期匹配步骤中使用哪种相机模型，默认为 ```OPENCV```。
  #### --resize
  用于创建输入图像的缩小版本的标志。
  #### --colmap_executable
  指向COLMAP可执行文件的路径（Windows上是 ```.bat``` 文件）。
  #### --magick_executable
  指向ImageMagick可执行文件的路径。
</details>
<br>

## 常见问题解答
- *我从哪里获取数据集，例如在 ```full_eval.py``` 中引用的数据集？* MipNeRF360数据集由原始论文的作者在项目网站上提供。请注意，其中两个数据集不能公开共享，需要您直接咨询作者。对于Tanks&Temples和Deep Blending，请使用页面顶部提供的下载链接。或者，您也可以从[HuggingFace](https://huggingface.co/camenduru/gaussian-splatting)获取克隆的数据（状态：2023年8月！）。

- *我如何使用这个方法处理更大的数据集，比如一个城市区域？* 当前的方法并没有为这些设计，但只要有足够的内存，它应该能够工作。然而，该方法在多尺度细节场景（极端特写与远景混合）中可能会遇到困难。这通常出现在，例如，驾驶数据集中（近距离的汽车，远距离的建筑物）。对于这样的场景，您可以降低 ```--position_lr_init```、```--position_lr_final``` 和 ```--scaling_lr```（x0.3, x0.1, ...）。场景越广泛，这些值应该越低。下面，我们使用默认的学习率（左）和 ```--position_lr_init 0.000016 --scaling_lr 0.001"```（右）。



| ![默认学习率结果](assets/worse.png "标题-1") <!-- --> | <!-- --> ![降低学习率结果](assets/better.png "标题-2") |
| --- | --- |

- *我用的是Windows系统，无法构建子模块，怎么办？* 可以考虑按照[这里](https://www.youtube.com/watch?v=UXtuigy_wYc)的优秀视频教程中的步骤进行操作，希望这能帮到您。执行步骤的顺序很重要！或者，您可以考虑使用链接中的Colab模板。

- *还是不行，它说有关于 ```cl.exe``` 的问题，我该怎么办？* 用户Henry Pearce找到了一个解决方案。您可以尝试将Visual Studio的路径添加到您的环境变量中（您的版本号可能有所不同）；
```C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64```
然后确保您启动一个新的conda提示符，然后转到您的仓库位置，尝试执行以下操作；
```
conda activate gaussian_splatting
cd <dir_to_repo>/gaussian-splatting
pip install submodules\diff-gaussian-rasterization
pip install submodules\simple-knn
```


- *我用的是macOS/Puppy Linux/Greenhat，无法进行构建，怎么办？* 很抱歉，我们无法为README中未列出的平台提供支持。您可以考虑使用链接中的Colab模板。

- *我没有24GB的显存用于训练，该怎么办？* 显存消耗取决于正在优化的点的数量，这个数量会随着时间的推移而增加。如果您只想训练到7k次迭代，您将需要明显更少的显存。为了进行完整的训练过程并避免内存不足，您可以增加 ```--densify_grad_threshold```、```--densification_interval``` 的值，或减小 ```--densify_until_iter``` 的值。但请注意，这将影响结果的质量。还可以尝试将 ```--test_iterations``` 设置为 ```-1```，以避免测试期间的内存峰值。如果 ```--densify_grad_threshold``` 设得非常高，那么应该不会发生密集化，如果场景本身成功加载，训练应该能够完成。

- *24GB的显存用于参考质量训练仍然很多！我们能用更少的显存完成吗？* 是的，很有可能。根据我们的计算，应该可以用**远**少于24GB的显存（大约8GB）完成训练。如果我们有时间，我们会尝试实现这一点。如果有PyTorch方面的高手愿意解决这个问题，我们期待您的拉取请求！

- *我如何将可微分的高斯光栅化器用于我的自己的项目？* 很简单，它包含在这个仓库的一个子模块 ```diff-gaussian-rasterization``` 中。请随意查看并安装该软件包。虽然没有详细的文档，但从Python端使用它非常直接（参见 ```gaussian_renderer/__init__.py```）。

- *等等，但是```<插入特性>```没有被优化，可能会更好吗？* 有几个部分我们甚至还没有时间去考虑改进（尚未）。您用这个原型获得的性能很可能是物理上可能的相当慢的基线。

- *出了问题，这是怎么回事？* 我们努力提供一个坚实和易于理解的基础，以便利用论文的方法。我们已经对代码进行了相当多的重构，但我们的测试能力有限，无法测试所有可能的使用场景。因此，如果网站、代码或性能存在问题，请创建一个问题。如果我们有时间，我们将尽力解决。
