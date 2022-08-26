# ErfNet SDK开发样例

## 1 介绍

ErfNet是一个语义分割网络，ERFNet可以看作是对ResNet结构的又一改变，ERFNet提出了Factorized Residual Layers，内部全部使用1D的cov(非对称卷积)，以此来降低参数量，提高速度。同时ERFNet也是对ENet的改进，在模型结构上删除了encode中的层和decode层之间的long-range链接，同时所有的downsampling模块都是一组并行的max pooling和conv。

本项目基于Mind SDK框架实现了ErfNet模型的推理。

### 1.1 支持的产品

Ascend 310

### 1.2 支持的版本

CANN：5.0.4
SDK：2.0.4

版本号查询方法，在Atlas产品环境下，运行命令：

```
npu-smi info
```

### 1.3 软件方案介绍

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | 模型转换系统    | 本系统包含实现了pth文件到om文件的转换 |
| 2    | SDK推理系统    | 本系统实现了基于SDK的模型推理 |

表1.2 模型转换系统系统方案中各模块功能：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | pth2om.sh    | 生成om模型的脚本 |
| 2    | modify_bn_weights.py    | 修正模型参数的脚本 |
| 3    | ErfNet_pth2onnx.py    | 生成onnx模型的脚本 |

表1.3 SDK推理系统系统方案中各模块功能：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | infer.py | 实现推理功能的脚本 |
| 2    | erfnet_pipeline.json   | SDK的推理pipeline配置文件 |
| 3    |  test_metric.py  | 测试模型精度的脚本 |

### 1.4 代码目录结构与说明

工程目录如下图所示：
```
|-- erfnet_pipeline.json    // SDK的推理pipeline配置文件
|-- erfnet_pretrained.pth   // 预训练的Pytorch模型
|-- ErfNet_pth2onnx.py      // 生成onnx模型的脚本 
|-- infer.py                // 实现推理功能的脚本           
|-- test_metric.py          // 测试模型精度的脚本           
|-- modify_bn_weights.py    // 修正模型参数的脚本                    
|-- pth2om.sh               // 生成om模型的脚本             
|-- README.md               // 自述文件           
```

### 1.5 技术实现流程

本项目首先通过onnx软件将pytorch的预训练模型转化为onnx模型，然后在使用atc工具将其转化为SDK能使用的om模型。最终我们通过构建SDK推理pipeline，实现模型推理。

### 1.6 特性及适用场景

ErfNet原论文使用街景图片来进行语义分割任务的测试，ErfNet的原理并没有根据具体场景设计，所以其他分割任务也都能使用。

## 2 环境依赖

环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| -------- | ------ |
| Pytorch  | 1.12.1 |
| PIL   |     9.0.1   |
| numpy         |    1.23.2    |

## 3准备工作

### 生成模型文件

pth权重文件下载参考华为昇腾社区[ModelZoo](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/a552b9d78220425f9a59f0ffdb083dfa)
将pth权重文件下载至项目根目录（pth权重文件来源于原作者的github仓库），在项目根目录下键入```bash pth2om.sh```，会在同目录下得到```ErfNet_bs1.om```文件。

### 下载数据集

[获取cityscapes](https://www.cityscapes-dataset.com/)
- Download the Cityscapes dataset from https://www.cityscapes-dataset.com/

  - Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels.
  - Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds  

## 4SDK推理
在项目根目录下键入

```bash
python infer.py --pipeline_path erfnet_pipeline.json  --data_path /datapath --output_path ./infer_result
```

其中参数```--pipeline_path```为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
```--data_path```参数为数据集的路径；```--output_path```参数是推理结果的输出路径。
最终用户可以在output_path路径下查看结果。

## 5测试精度

在项目根目录下键入

```bash
python test_metric.py --pipeline_path erfnet_pipeline.json  --data_path /path/to/cityscapes/
```

其中参数```--pipeline_path```为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
```--data_path```参数为数据集的路径。待脚本运行完毕，会输出以下信息。

```
mean_iou:  0.7219515597801778
iou_class:  tensor([0.9762, 0.8137, 0.9078, 0.4939, 0.5493, 0.6080, 0.6262, 0.7231, 0.9135,
        0.6096, 0.9339, 0.7612, 0.5345, 0.9291, 0.7274, 0.7886, 0.6375, 0.4646,
        0.7190], dtype=torch.float64)
```

<!-- 
## 5 软件依赖说明

如果涉及第三方软件依赖，请详细列出。

| 依赖软件 | 版本  | 说明                     |
| -------- | ----- | ------------------------ |
| Pytorch      | 1.12.1 | 用于计算指标 |
|          |       |                          | -->


<!-- ## 6 常见问题

请按照问题重要程度，详细列出可能要到的问题，和解决方法。

### 6.1 XXX问题

**问题描述：**

截图或报错信息

**解决方案：**

详细描述解决方法。 -->