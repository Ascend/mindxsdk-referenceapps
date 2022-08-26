# PraNet SDK开发样例

## 1 介绍

PraNet是一种针对息肉分割任务需求设计的，名为并行反向注意力的深度神经网络。
基于并行反向注意力的息肉分割网络（PraNet），利用并行的部分解码器（PPD）在高级层中聚合特征作为初始引导区域，
再使用反向注意模块（RA）挖掘边界线索。

本项目基于Mind SDK框架实现了PraNet模型的推理。

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
| 2    | PraNet_pth2onnx.py    | 生成onnx模型的脚本 |

表1.3 SDK推理系统系统方案中各模块功能：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | infer.py | 实现推理功能的脚本 |
| 2    | pranet_pipeline.json   | SDK的推理pipeline配置文件 |
| 3    |  test_metric.py  | 测试模型精度的脚本 |

### 1.4 代码目录结构与说明

工程目录如下图所示：
```
|-- pranet_pipeline.json    // SDK的推理pipeline配置文件
|-- PraNet_pth2onnx.py      // 生成onnx模型的脚本 
|-- infer.py                // 实现推理功能的脚本           
|-- test_metric.py          // 测试模型精度的脚本           
|-- pth2om.sh               // 生成om模型的脚本             
|-- README.md               // 自述文件           
```

### 1.5 技术实现流程图

本项目首先通过onnx软件将pytorch的预训练模型转化为onnx模型，然后在使用atc工具将其转化为SDK能使用的om模型。最终通过构建SDK推理pipeline，实现模型推理。

### 1.6 特性及适用场景

在医疗图像处理领域，PraNet针对息肉识别需求而设计。

## 2 环境依赖

环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| -------- | ------ |
| Pytorch  | 1.12.1 |
| PIL   |     9.0.1   |
| numpy         |    1.23.2    |
| onnx         |    1.12.0    |
| onnxsim         |    0.4.7    |

## 3准备工作

### 生成模型文件

pth权重文件下载参考华为昇腾社区[ModelZoo](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/e08e0552334ec81d8e632fafbb22a9f0)
将pth权重文件下载至项目根目录（pth权重文件来源于原作者的github仓库），在项目根目录下键入```bash pth2om.sh```，会在同目录下得到```PraNet-19_bs1.om```文件。

### 下载数据集

本模型支持Kvasir的验证集。请用户需自行获取Kvasir数据集，上传数据集到服务器任意目录并解压（如：/root/datasets）。

```
Kvasir
├── images
├── masks
```

## 4SDK推理
在项目根目录下键入

```bash
python infer.py --pipeline_path pranet_pipeline.json --data_path /path/to/images --output_path ./infer_result
```

其中参数```--pipeline_path```为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
```--data_path```参数为数据集的路径；```--output_path```参数是推理结果的输出路径。
最终用户可以在output_path路径下查看结果。

## 5测试精度

在项目根目录下键入

```bash
python test_metric.py --pipeline_path pranet_pipeline.json --data_path /path/to/Kvasir
```

其中参数```--pipeline_path```为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
```--data_path```参数为数据集的路径。待脚本运行完毕，会输出以下精度信息。

```
dataset      meanDic    meanIoU
---------  ---------  ---------
res            0.895      0.836
#################### End Evaluation ####################
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