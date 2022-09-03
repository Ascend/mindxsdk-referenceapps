# ErfNet SDK开发样例

## 1 介绍

ErfNet是一个语义分割网络，ERFNet可以看作是对ResNet结构的又一改变，ERFNet提出了Factorized Residual Layers，内部全部使用1D的cov(非对称卷积)，以此来降低参数量，提高速度。同时ERFNet也是对ENet的改进，在模型结构上删除了encode中的层和decode层之间的long-range链接，同时所有的downsampling模块都是一组并行的max pooling和conv。

本项目基于Mind SDK框架实现了ErfNet模型的推理。

### 1.1 支持的产品

Ascend 310

### 1.2 支持的版本

CANN：5.0.4（通过cat /usr/local/Ascend/ascend-toolkit/latest/acllib/version.info，获取版本信息）

SDK：2.0.4（可通过cat SDK目录下的version.info查看信息）

### 1.3 代码目录结构与说明

工程目录如下图所示：

```
|-- pipeline
|   |-- erfnet_pipeline.json // SDK的推理pipeline配置文件
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
|-- env.sh                    // 环境配置
```

### 1.4 技术实现流程

本项目首先使用atc工具将其转化为SDK能使用的om模型。最终我们通过构建SDK推理pipeline，实现模型推理。

### 1.5 特性及适用场景

ErfNet原论文使用街景图片来进行语义分割任务的测试，ErfNet的原理并没有根据具体场景设计，所以其他分割任务也都能使用。

## 2 环境依赖

推荐系统为ubuntu 18.04，环境依赖软件和版本如下表：

| 软件名称            | 版本        | 说明                          | 获取方式                                                     |
| ------------------- | ----------- | ----------------------------- | ------------------------------------------------------------ |
| MindX SDK           | 2.0.4       | mxVision软件包                | [链接](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2FMindx-sdk) |
| ubuntu              | 18.04.1 LTS | 操作系统                      | Ubuntu官网获取                                               |
| Ascend-CANN-toolkit | 5.0.4       | Ascend-cann-toolkit开发套件包 | [链接](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2Fcann%2Fcommercial) |
| python              | 3.9.2       |                               |                                                              |
| numpy               | 1.22.4      | 维度数组运算依赖库            | 服务器中使用pip或conda安装                                   |
| PIL       | 9.0.1       | 图像处理依赖库                | 服务器中使用pip或conda安装                                   |
| opencv-python       | 4.6.0       | 图像处理依赖库                | 服务器中使用pip或conda安装                                   |

## 准备

### 获取OM模型文件

OM权重文件获取参考华为昇腾社区[ModelZoo](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/a552b9d78220425f9a59f0ffdb083dfa)。
获取到```ErfNet.onnx```模型后，将其放在model目录下。首先在根目录执行```source env.sh```激活环境；然后在model目录键入以下命令

```
bash onnx2om.sh
```

能获得```ErfNet_bs1.om```模型文件。

注: [ModelZoo](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/a552b9d78220425f9a59f0ffdb083dfa)
中的模型文件```ErfNet_bs1.om```不能用于本项目。

### 编译插件

首先进入文件夹```plugin/postprocess/```，键入```bash build.sh```，对后处理插件进行编译。

### 下载数据集

[获取cityscapes](https://www.cityscapes-dataset.com/)
* Download the Cityscapes dataset from https://www.cityscapes-dataset.com/

  + 下载leftImg8bit.zip以获得RGB图片, 下载gtFine.zip以获得标签.
  + 应使用的标签为"_labelTrainIds"而非"_labelIds", 你可以下载[cityscapes scripts](https://github.com/mcordts/cityscapesScripts)并使用[conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py)来生成trainIds。

### 创建推理文件目录

在项目根目录下键入

```
mkdir infer_result
```

以创建推理输出目录

## 4SDK推理

在项目根目录下键入

```bash
python main.py --pipeline_path ./pipeline/erfnet_pipeline.json  --data_path ./data/
```

其中参数` ` ` --pipeline_path ` ` `为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
` ` ` --data_path ` `  `参数为数据所在文件夹的路径，本项目提供了一个样例图片位于```./data/```目录下；
最终用户可以在项目根目录下找到推理结果```testout.jpg```图片。

## 5测试精度


在项目根目录下键入

```bash
python test_metric.py --pipeline_path ./pipeline/erfnet_pipeline.json --data_path /path/to/cityscapes/
```

其中参数` ` ` --pipeline_path ` ` `为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
` ` ` --data_path ` ` `参数为数据集的路径。待脚本运行完毕，会输出以下信息。

```
mean_iou:  0.7219515597801778
iou_class:  tensor([0.9762, 0.8137, 0.9078, 0.4939, 0.5493, 0.6080, 0.6262, 0.7231, 0.9135,
        0.6096, 0.9339, 0.7612, 0.5345, 0.9291, 0.7274, 0.7886, 0.6375, 0.4646,
        0.7190], dtype=float64)
```

