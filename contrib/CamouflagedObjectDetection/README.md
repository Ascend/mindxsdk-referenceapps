# MindX SDK -- 伪装目标分割参考设计案例

## 1 案例概述

### 1.1 概要描述

在本系统中，目的是基于MindX SDK，在华为云昇腾平台上，开发端到端**伪装目标分割**的参考设计，实现**对图像中的伪装目标进行识别检测**的功能，达到功能要求

### 1.2 模型介绍

本项目主要基于用于通用伪装目标分割任务的DGNet模型

- 模型的具体描述和细节可以参考原文：https://arxiv.org/abs/2205.12853

- 具体实现细节可以参考基于PyTorch深度学习框架的代码：https://github.com/GewelsJI/DGNet/tree/main/lib_pytorch

- 所使用的公开数据集是NC4K，可以在此处下载：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/snapshots/data.tar

- 所使用的模型是EfficientNet-B4版本的DGNet模型，原始的PyTorch模型文件可以在此处下载：

### 1.3 实现流程

- 基础环境：Ascend 310、mxVision、Ascend-CANN-toolkit、Ascend Driver
- 模型转换：将ONNX模型（.onnx）转换为昇腾离线模型（.om）
- 昇腾离线模型推理流程代码开发

### 1.4 代码地址

本项目的代码地址为：https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/contrib/CamouflagedObjectDetection

## 2 软件方案介绍

### 2.1 项目方案架构介绍

本系统设计了不同的功能模块。主要流程为：图片传入流中，利用DGNet检测模型检测伪装目标，将检测出的伪装目标以逐像素概率图的形式输出，各模块功能描述如表2.1所示：

表2.1 系统方案中各模块功能：

| 序号 | 子系统 | 功能描述 |
| :------------ | :---------- | :---------- |
| 1    | 图像输入 | 调用PIL.Image中的图片加载函数，用于加载输入图片|
| 2    | 图像前处理 | 调用torchvision.transforms函数，将输入图片放缩到352*352大小，并执行归一化操作 |
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
└── utils
    ├── dataset.py  # pytorch的dataloader文件
    ├── __init__.py
    └── inference_onnx.py # ONNX离线模型推理python脚本文件
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
| opencv-python |    4.5.3.56    |

### 3.2 环境搭建

#### 3.2.1 基础环境变量设置（ATC转换工具）

ATC工具介绍请参考：https://support.huawei.com/enterprise/zh/doc/EDOC1100234054/83f40ac7

```bash
# 执行如下命令，打开.bashrc文件
vim ~/.bashrc

# 在.bashrc文件中添加以下环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:$LD_LIBRARY_PATH
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH
export PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:$PATH
export ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
export ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
export TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

# 保存退出.bashrc文件，并执行如下命令使环境变量生效
source ~/.bashrc

# 查看环境变量并检查是否正确添加
env
```

#### 3.2.2 昇腾离线模型推理环境变量设置（MindX SDK工具）

请参考此链接安装MindX SDK工具包：https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/quick_start/1-1%E5%AE%89%E8%A3%85SDK%E5%BC%80%E5%8F%91%E5%A5%97%E4%BB%B6.md

```bash

# 执行如下命令，打开.bashrc文件
vim ~/mindx_dir/mxVision/set_env.sh


# 在set_env.sh中添加如下设置
path="${BASH_SOURCE[0]}"

if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
  sdk_path=$(cd $(dirname $path); pwd )

  if [[ -f "$sdk_path"/filelist.txt ]] && [[ -f "$sdk_path"/version.info ]];then
    export MX_SDK_HOME="$sdk_path"
    export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
    export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0":"${MX_SDK_HOME}/lib/plugins"
    export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/modelpostprocessors":"${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":${LD_LIBRARY_PATH}
    export PYTHONPATH=${MX_SDK_HOME}/python:$PYTHONPATH
  else
    echo "The package is incomplete, please check it."
  fi
else
  echo "There is no 'set_env.sh' to import"
fi
```

### 3.3 模型转换

**步骤1** 在GitHub上下载DGNet (Efficient-B4) 的ONNX模型，下载地址为：https://github.com/GewelsJI/DGNet/releases/download/Checkpoints/DGNet.pth

**步骤2** 将获取到的DGNet模型onxx文件存放至`./snapshots/DGNet/DGNet.onnx`。

**步骤3** 模型转换具体步骤

```bash
# 请确保之前完成 3.2.1 节中基础环境变量设置，这样能够确保ATC转换工具的正常使用（如果已经执行，请跳过此步骤）
source ~/.bashrc
# 进入对应目录
cd ./snapshots/DGNet/
# 执行以下命令将ONNX模型（.onnx）转换为昇腾离线模型（.om）
atc --framework=5 --model=DGNet.onnx --output=DGNet --input_shape="image:1,3,352,352" --log=debug --soc_version=Ascend310 > atc.log
```

执行完模型转换脚本后，会在对应目录中获取到如下转化模型：DGNet.om（本项目中在Ascend平台上所使用的离线模型文件）。


## 4 推理与评测

示例步骤如下：

**步骤0** 

参考1.2节中说明下载一份测试数据集合：下载链接：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/snapshots/data.tar

**步骤1** 

请确保之前完成 3.2.2 节中昇腾离线模型推理环境变量设置，这样能够确保MindX SDK工具的正常使用（如果已经执行，请跳过此步骤）

```bash
. ~/mindx_dir/mxVision/set_env.sh
```

**步骤2** 

执行离线推理Python脚本

```bash
python ./inference_om.py
```

**步骤3**

- 定量性能验证：使用原始GitHub仓库中提供的[标准评测代码](https://github.com/GewelsJI/DGNet/blob/main/lib_ascend/evaluation.py)进行测评，然后可以生成评测指标数值表格。可以看出DGNet模型的Smeasure指标数值为0.84，超过了项目交付中提到的“大于0.84”的要求。

  ```text
  +---------+--------------+----------+
  | Dataset |    Method    | Smeasure |
  +---------+--------------+----------+
  |   NC4K  | Exp-DGNet-OM |  0.84    |
  +---------+--------------+----------+
  ```

- 定性性能验证：
  - 输入伪装图片：![](./assets/74.jpg)
  - 预测分割结果：![](./assets/74.png)

## 5 参考引用

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
