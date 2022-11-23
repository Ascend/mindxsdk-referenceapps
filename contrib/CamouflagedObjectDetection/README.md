# MindX SDK -- 伪装目标分割参考设计案例

## 1 案例概述

### 1.1 概要描述

在本系统中，目的是基于MindX SDK，在华为云昇腾平台上，开发端到端**伪装目标分割**的参考设计，实现**对图像中的伪装目标进行识别检测**的功能，达到功能要求

### 1.2 模型介绍

本项目主要基于用于通用伪装目标分割任务的DGNet模型

- 模型的具体描述和细节可以参考原文：https://arxiv.org/abs/2205.12853

- 具体实现细节可以参考基于PyTorch深度学习框架的代码：https://github.com/GewelsJI/DGNet/tree/main/lib_pytorch

- 所使用的公开数据集是NC4K，可以在此处下载：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/snapshots/data.tar

- 所使用的模型是EfficientNet-B4版本的DGNet模型，原始的PyTorch模型文件可以在此处下载：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/snapshots/DGNet.zip


### 1.3 实现流程

- 基础环境：Ascend 310、mxVision、Ascend-CANN-toolkit、Ascend Driver
- 模型转换：将ONNX模型（.onnx）转换为昇腾离线模型（.om）
- 昇腾离线模型推理流程代码开发

### 1.4 代码地址

本项目的代码地址为：https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/contrib/CamouflagedObjectDetection

### 1.5 特性及适用场景

本项目适用于自然场景下图片完整清晰、无模糊鬼影的场景，并且建议输入图片为JPEG编码格式，大小不超过10M。

**注意：由于模型限制，本项目暂只支持自然场景下伪装动物的检测，不能用于其他用途**

## 2 软件方案介绍

### 2.1 项目方案架构介绍

本系统设计了不同的功能模块。主要流程为：图片传入流中，利用DGNet检测模型检测伪装目标，将检测出的伪装目标以逐像素概率图的形式输出，各模块功能描述如表2.1所示：

表2.1 系统方案中各模块功能：

| 序号 | 子系统 | 功能描述 |
| :------------ | :---------- | :---------- |
| 1    | 图像输入 | 调用cv2中的图片加载函数，用于加载输入图片|
| 2    | 图像前处理 | 将输入图片放缩到352*352大小，并执行归一化操作 |
| 3    | 伪装目标检测 | 利用DGNet检测模型，检测出图片中的伪装目标|
| 4    | 数据分发 | 将DGNet模型检测到的逐像素概率图进行数据分发到下个插件|
| 5    | 结果输出 | 将伪装目标概率预测图结果进行输出并保存|

### 2.2 代码目录结构与说明

本工程名称为DGNet，工程目录如下列表所示：

```
./
├── assets  # 文件
│   ├── 74.jpg
│   └── 74.png
├── data  # 数据集存放路径
│   └── NC4K
├── inference_om.py # 昇腾离线模型推理python脚本文件
├── README.md # 本文件
├── seg_results_om
│   ├── Exp-DGNet-OM  # 预测结果图存放路径
├── snapshots
│   ├── DGNet  # 模型文件存放路径
```

## 3 开发准备

### 3.1 环境依赖说明

环境依赖软件和版本如下表：

|   软件名称    |    版本     |
| :-----------: | :---------: |
|    ubantu     | 18.04.1 LTS |
|   MindX SDK   |    2.0.4    |
|    Python     |    3.9.2    |
|     CANN      |    5.0.4    |
|     numpy     |   1.21.2    |
| opencv-python |  4.5.3.56   |
| mindspore (cpu) |     1.9.0   |

### 3.2 环境搭建

在编译运行项目前，需要设置环境变量

```bash
# MindXSDK 环境变量：
. ${SDK-path}/set_env.sh

# CANN 环境变量：
. ${ascend-toolkit-path}/set_env.sh

# 环境变量介绍
SDK-path: SDK mxVision 安装路径
ascend-toolkit-path: CANN 安装路径
```

### 3.3 模型转换

**步骤1** 下载DGNet (Efficient-B4) 的ONNX模型：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/snapshots/DGNet.zip

**步骤2** 将下载获取到的DGNet模型onxx文件存放至`./snapshots/DGNet/DGNet.onnx`。

**步骤3** 模型转换具体步骤

```bash
# 进入对应目录
cd ./snapshots/DGNet/
# 执行以下命令将ONNX模型（.onnx）转换为昇腾离线模型（.om）
atc --framework=5 --model=DGNet.onnx --output=DGNet --input_shape="image:1,3,352,352" --log=debug --soc_version=Ascend310
```

执行完模型转换脚本后，会在对应目录中获取到如下转化模型：DGNet.om（本项目中在Ascend平台上所使用的离线模型文件）。

## 4 推理与评测

示例步骤如下：

**步骤0** 

参考1.2节中说明下载一份测试数据集合：下载链接：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/snapshots/data.tar

**步骤1** 

执行离线推理Python脚本

```bash
python inference_om.py --om_path ./snapshots/DGNet/DGNet.om --save_path ./seg_results_om/Exp-DGNet-OM/NC4K/ --data_path ./data/NC4K/Imgs 
```

**步骤2** 

- 定量性能验证：

使用原始GitHub仓库中提供的标准评测代码进行测评，具体操作步骤如下：

```bash
# 拉取原始仓库
git clone https://github.com/GewelsJI/DGNet.git

# 将如下两个文件夹放入当前
mv ./DGNet/lib_ascend/eval ./contrib/CamouflagedObjectDetection/
mv ./DGNet/lib_ascend/evaluation.py ./contrib/CamouflagedObjectDetection/

# 运行如下命令进行测评
python evaluation.py
```

然后可以生成评测指标数值表格。可以看出DGNet模型的Smeasure指标数值为0.856，已经超过了项目交付中提到的“大于0.84”的要求。

```text
+---------+-----------------------+----------+-----------+-------+-------+--------+-------+-------+--------+-------+
| Dataset |         Method        | Smeasure | wFmeasure |  MAE  | adpEm | meanEm | maxEm | adpFm | meanFm | maxFm |
+---------+-----------------------+----------+-----------+-------+-------+--------+-------+-------+--------+-------+
|   NC4K  |      Exp-DGNet-OM     |  0.856   |   0.782   | 0.043 | 0.909 |  0.91  | 0.921 |  0.8  | 0.812  | 0.833 |
+---------+-----------------------+----------+-----------+-------+-------+--------+-------+-------+--------+-------+
```

- 定性性能验证：

输入伪装图片：![](./assets/74.jpg)
预测分割结果：![](./assets/74.png)

## 5 参考引用

主要参考为如下三篇论文：

    @article{ji2022gradient,
      title={Deep Gradient Learning for Efficient Camouflaged Object Detection},
      author={Ji, Ge-Peng and Fan, Deng-Ping and Chou, Yu-Cheng and Dai, Dengxin and Liniger, Alexander and Van Gool, Luc},
      journal={Machine Intelligence Research},
      year={2023}
    } 

    @article{fan2021concealed,
      title={Concealed Object Detection},
      author={Fan, Deng-Ping and Ji, Ge-Peng and Cheng, Ming-Ming and Shao, Ling},
      journal={IEEE TPAMI},
      year={2022}
    }

    @inproceedings{fan2020camouflaged,
      title={Camouflaged object detection},
      author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
      booktitle={IEEE CVPR},
      pages={2777--2787},
      year={2020}
    }
