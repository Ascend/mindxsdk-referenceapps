# ErfNet SDK开发样例

## 1 介绍

ErfNet是一个语义分割网络，ERFNet可以看作是对ResNet结构的又一改变，ERFNet提出了Factorized Residual Layers，内部全部使用1D的cov(非对称卷积)，以此来降低参数量，提高速度。同时ERFNet也是对ENet的改进，在模型结构上删除了encode中的层和decode层之间的long-range链接，同时所有的downsampling模块都是一组并行的max pooling和conv。

本项目基于Mind SDK框架实现了ErfNet模型的推理。

### 1.1 支持的产品

本项目以昇腾Atlas 500 A2为主要的硬件平台。

### 1.2 支持的版本

| 软件名称 | 版本   |
| -------- | ------ |
| python    | 3.9.2     | 
| MindX SDK     |    5.0RC1    |
| CANN | 310使用6.3.RC1<br>310B使用6.2.RC1 |

### 1.3 代码目录结构与说明

工程目录如下图所示：

```
|-- pipeline
|   |-- erfnet_pipeline.pipeline // SDK的推理pipeline配置文件
|-- plugin
|   |-- postprocess
|       |-- build.sh          // 编译脚本
|       |-- CMakeLists.txt    // CMakeLists
|       |-- Postprocess.cpp   // 插件.cpp文件
|       |-- Postprocess.h     // 插件.h文件
|-- model
|   |-- erfnet.aippconfig     // aippconfig
|   |-- onnx2om.sh            // 模型转换脚本
|-- main.py                   // 实现推理功能的脚本
|-- test_metric.py            // 测试模型精度的脚本
|-- README.md                 // 自述文件
```

### 1.4 技术实现流程

本项目首先使用atc工具将其转化为SDK能使用的om模型。最终我们通过构建SDK推理pipeline，实现模型推理。

### 1.5 特性及适用场景

ErfNet原论文使用街景图片来进行语义分割任务的测试，ErfNet的原理并没有根据具体场景设计，所以其他分割任务也都能使用。

在CityScapes数据集上，该项目精度能够达到原论文的水平。不过在一些场景上的效果还不够理想:

+ 1. 当图片来源于车载录像，图片中会包含本车的一部分，网络对这部分的分割结果有所缺陷，表现为，轮廓模糊，分类不准确等。
+ 2. 图片中场景过于复杂时，街道和物体交织在一起，网络对这类场景的分割结果也不够理想。
+ 3. 轻微的精度损失：该模型相比于原模型精度稍有下降，这是因为mindsdk只提供了jpg格式图片的解码，而原数据集中的图片为png格式，所以为了将模型迁移到mindsdk，需要将数据全部转换为jpg格式。而jpg格式压缩图片是有损失的，所以造成了一定的精度下降。


## 2 环境依赖

推荐系统为ubuntu 18.04

| 软件名称            | 版本        | 说明                          | 获取方式                                                     |
| ------------------- | ----------- | ----------------------------- | ------------------------------------------------------------ 
| numpy               | 1.22.4      | 维度数组运算依赖库            | 服务器中使用pip或conda安装                                   |
| PIL       | 9.0.1       | 图像处理依赖库                | 服务器中使用pip或conda安装                                   |
| opencv-python       | 4.6.0       | 图像处理依赖库                | 服务器中使用pip或conda安装                                   |
| pyquaternion | | |服务器中使用pip或conda安装 |

> 配置环境变量。

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh #toolkit默认安装路径，根据实际安装路径修改
. ${SDK_INSTALL_PATH}/mxVision/set_env.sh
```

## 3 准备

### 3.1 获取OM模型文件

OM权重文件获取参考华为昇腾社区[ModelZoo](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ErfnetSegementation/ATC%20ErfNet%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip)。
获取到```ErfNet.onnx```模型后，将其放在model目录下。在model目录键入以下命令

```
bash onnx2om.sh
```

能获得```ErfNet_bs1.om```模型文件。

注: [ModelZoo](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ErfnetSegementation/ATC%20ErfNet%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip)
中的模型文件```ErfNet_bs1.om```不能用于本项目。

### 3.2 编译插件

首先进入文件夹```plugin/postprocess/```，键入```bash build.sh```，对后处理插件进行编译。

### 3.3 下载数据集

[获取cityscapes](https://www.cityscapes-dataset.com/)
* Download the Cityscapes dataset from https://www.cityscapes-dataset.com/

  + 下载leftImg8bit.zip以获得RGB图片, 下载gtFine.zip以获得标签.
  + 应使用的标签为"_labelTrainIds"而非"_labelIds", 你可以下载[cityscapes scripts](https://github.com/mcordts/cityscapesScripts)并使用[conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py)来生成trainIds。

在官网上下载数据集：gtFine_trainvaltest.zip (241MB) , leftImg8bit_trainvaltest.zip (11GB).

在根目录下创建文件夹cityscapes，将数据集解压到cityscapes下，得到以下目录：

```
cityscapes
|  └── gtFine
|  └── leftImg8bit
```

原数据集标签有34类别，而我们需要使用19类，为了应用数据集，需要对标签文件进行转换。首先键入

```bash
git clone https://github.com/mcordts/cityscapesScripts.git
```

在cityscapesscripts/preparation/createTrainIdLabelImgs.py脚本中

```
import os, glob, sys 
```

此行后增加如下2行

```
import pdb
sys.path.append('cityscapesScripts的绝对路径')
```

下载cityscapes数据集工具包，然后键入

```bash
export CITYSCAPES_DATASET="cityscapes文件夹的绝对路径"
```

环境变量CITYSCAPES_DATASET用于标识数据集的位置，键入

```bash
cd cityscapesScripts
python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

出现以下标识，说明转换成功。

```bash
Processing 500 annotation files
Progress: 100.0 % 
```

## 4 SDK推理

在项目根目录下键入

```bash
python main.py --pipeline_path ./pipeline/erfnet_pipeline.pipeline  --data_path ./data/
```

其中参数` ` ` --pipeline_path ` ` `为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
` ` ` --data_path ` `  `参数为数据所在文件夹的路径，本项目提供了一个样例图片位于```./data/```目录下；
最终用户可以在项目根目录下找到推理结果```./infer_result/```图片。

## 5 测试精度


在项目根目录下键入

```bash
python test_metric.py --pipeline_path ./pipeline/erfnet_pipeline.pipeline --data_path ./cityscapes/
```

其中参数` ` ` --pipeline_path ` ` `为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
` ` ` --data_path ` ` `参数为数据集的路径。待脚本运行完毕，会输出以下信息。

```
mean_iou:  0.704181969165802
iou_class:  [0.9736657  0.7985532  0.90132016 0.43224114 0.5292813  0.5920671
 0.59510547 0.7050008  0.90608126 0.5688207  0.9345611  0.75305563
 0.5223323  0.9229183  0.68231714 0.7712575  0.6587059  0.43045282
 0.7017194 ]
```

目标精度为原论文所达到的精度，为68.0%。该项目达到的指标为70.4%，超过了目标精度。
