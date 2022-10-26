# PraNet SDK开发样例

## 1 介绍

PraNet是一种针对息肉分割任务需求设计的，名为并行反向注意力的深度神经网络。
基于并行反向注意力的息肉分割网络（PraNet），利用并行的部分解码器（PPD）在高级层中聚合特征作为初始引导区域，
再使用反向注意模块（RA）挖掘边界线索。

本项目基于Mind SDK框架实现了PraNet模型的推理。

### 1.1 支持的产品

Ascend 310

### 1.2 支持的版本

CANN：5.0.4（通过cat /usr/local/Ascend/ascend-toolkit/latest/acllib/version.info，获取版本信息）

SDK：2.0.4（可通过cat SDK目录下的version.info查看信息）

### 1.3 代码目录结构与说明

工程目录如下图所示：

```
|-- pipeline
|   |-- pranet_pipeline.json // SDK的推理pipeline配置文件
|-- plugin
|   |-- postprocess
|       |-- build.sh          // 编译脚本
|       |-- CMakeLists.txt    // CMakeLists
|       |-- Postprocess.cpp   // 插件.cpp文件
|       |-- Postprocess.h     // 插件.h文件
|-- model
|   |-- pranet.aippconfig     // aippconfig
|   |-- onnx2om.sh            // 模型转换脚本
|-- main.py                   // 实现推理功能的脚本
|-- test_metric.py            // 测试模型精度的脚本
|-- README.md                 // 自述文
```

### 1.4 技术实现流程图

本项目首先通过onnx软件将pytorch的预训练模型转化为onnx模型，然后在使用atc工具将其转化为SDK能使用的om模型。最终通过构建SDK推理pipeline，实现模型推理。

### 1.5 特性及适用场景

在医疗图像处理领域，PraNet针对息肉识别需求而设计。Pranet网络能够对息肉图片进行语义分割，功能正常，且精度达标。但是在以下情况下，分割效果不够理想：

1、当息肉相比整张图片面积很小时，分割效果不够理想，边缘会比较模糊。

2、当息肉大面具处于整张图片的边缘时，有一定概率分割失败，效果较差。(测试用例3.1.2)

轻微的精度损失：

该模型相比于原模型精度稍有下降，这是因为mindsdk只提供了jpg格式图片的解码，而原数据集中的图片为png格式，所以为了将模型迁移到mindsdk，需要将数据全部转换为jpg格式。而jpg格式压缩图片是有损失的，所以造成了一定的精度下降。

## 2 环境依赖

推荐系统为ubuntu 18.04，环境依赖软件和版本如下表：

| 软件名称            | 版本        | 说明                          | 获取方式                                                     |
| ------------------- | ----------- | ----------------------------- | ------------------------------------------------------------ |
| MindX SDK           | 2.0.4       | mxVision软件包                | [链接](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2FMindx-sdk) |
| ubuntu              | 18.04.1 LTS | 操作系统                      | Ubuntu官网获取                                               |
| Ascend-CANN-toolkit | 5.0.4       | Ascend-cann-toolkit开发套件包 | [链接](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2Fcann%2Fcommercial) |
| python              | 3.9.2       |                               |                                                              |
| numpy               | 1.22.4      | 维度数组运算依赖库            | 服务器中使用pip或conda安装                                   |
| opencv-python       | 4.6.0       | 图像处理依赖库                | 服务器中使用pip或conda安装                                   |
| PIL       | 9.0.1       | 图像处理依赖库                | 服务器中使用pip或conda安装                                   |
| onnx         |    1.12.0    | 模型转化库                | 服务器中使用pip或conda安装                                   |
| tabulate         |    0.8.10    | 格式化输出                | 服务器中使用pip或conda安装                                   |


键入

```bash
source ${SDK−path}/set_env.sh
source ${ascend-toolkit-path}/set_env.sh
```

以激活环境，其中```SDK-path```是SDK mxVision安装路径，```ascend-toolkit-path```是CANN安装路径。

## 3 准备工作

### 3.1 获取OM模型文件

OM权重文件下载参考华为昇腾社区[ModelZoo](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/e08e0552334ec81d8e632fafbb22a9f0)
获取到```PraNet-19.onnx```模型后，将其放在model目录下。在model目录键入以下命令

```
bash onnx2om.sh
```

能获得```PraNet-19_bs1.om```模型文件。

注: [ModelZoo](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/e08e0552334ec81d8e632fafbb22a9f0)
中的模型文件```PraNet-19_bs1.om```不能用于本项目。

### 3.2 编译插件

首先进入文件夹```plugin/postprocess/```，键入```bash build.sh```，对后处理插件进行编译。

### 3.3 下载数据集

数据集下载地址:
https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view

本模型使用Kvasir的验证集。请用户需自行获取Kvasir数据集，上传数据集到项目根目录并解压（如：/root/datasets）。

```
TestDataset
├── Kvasir
    ├── images
    ├── masks
```

## 4 SDK推理

在项目根目录下键入

```bash
python main.py --pipeline_path pipeline/pranet_pipeline.json --data_path ./TestDataset/Kvasir/images/
```

其中参数``` --pipeline_path ```为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
``` --data_path ```参数为推理图片的路径。最终用户可以在```./infer_result/```路径下查看结果。

## 5 测试精度

在项目根目录下键入

```bash
python test_metric.py --pipeline_path pipeline/pranet_pipeline.json --data_path ./TestDataset/Kvasir/
```

其中参数```--pipeline_path```为pipeline配置文件的路径，项目中已经给出该文件，所以直接使用相对路径即可；
```--data_path```参数为数据集的路径。待脚本运行完毕，会输出以下精度信息。

```
dataset      meanDic    meanIoU
---------  ---------  ---------
res            0.890      0.828
```
 
原模型的精度为：

```
dataset      meanDic    meanIoU
---------  ---------  ---------
res            0.895      0.836
```
 
精度符合要求。
