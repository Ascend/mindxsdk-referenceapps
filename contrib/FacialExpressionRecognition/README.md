# 情绪识别

## 1. 介绍

文本分类插件基于 MindXSDK 开发，在晟腾芯片上进行情绪识别，将分类结果在图片上显示。输入一张人脸图片，可以判断属于哪个情绪类别。
该模型支持7个类别：anger, disgust, fear, happy, sad, surprised, normal。

### 1.1 支持的产品

本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.2 支持的版本

支持的SDK版本为2.0.2。

### 1.3 软件方案介绍

基于MindX SDK的情绪识别业务流程为：待识别的图片通过open函数和read函数组成字符串，将字符串通过 appsrc 插件输入，
接着由媒体数据处理插件mxpi_imagedecoder和mxpi_imageresize对输入的图片进行解码和缩放操作，然后由推理模型插件mxpi_tensorinfer和mxpi_objectpostprocessor推理每张图片的人脸区域，
再通过媒体数据处理插件mxpi_imagecrop和mxpi_imageresize对输入的图片进行裁剪和缩放操作，最后通过推理模型插件mxpi_tensorinfer得到每种类别的得分。本系统的各模块及功能描述如表1所示：


表1.1 系统方案各子系统功能描述：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | 图片输入    | 读取输入文本 |
| 2    | 图片预处理    | 对输入的图片进行解码和缩放 |
| 3    | 人脸检测    | 推理出图片中的人脸区域 |
| 4    | 人脸区域抠图      | 从输入图片中裁剪出人脸区域 |
| 5    | 情绪识别    | 推理人脸的情绪类别|
| 6    | 情绪识别后处理    | 从模型推理结果中寻找对应的情绪标签|
| 7    | 输出结果    | 将推理结果输出|

### 1.4 代码目录结构与说明

本工程名称为文本分类，工程目录如下图所示：  

```
    │  build.sh
    │  label.txt
    │  main.py
    │  README.md
    │  run.sh
    │
    ├─image
    │     │ .keep
    │     │ test1.jpg
    │     │ test2.jpg
    │     │ test3.jpg
    │     │ test4.jpg
    │
    ├─model
    │     │ coco.names
    │     │ libyolov3postprocess.so
    │     │ scn.om
    │     │ yolov4.cfg
    │     │ yolov4_detection.om
    │
    ├─pipeline
    │     │ facial_expression_recognition.pipeline
    │     │ test.pipeline
    │
    └─test
          │  run.sh
          │  test.py
```
### 1.5 技术实现流程图
   ![Image text](image/FER.png)
本项目实现自然场景下的情绪识别，首先利用人脸检测模型采集图片中的人脸图像，然后利用情绪识别模型推理情绪类别。整体流程如下图所示。

## 2 环境依赖

推荐系统为ubantu 18.04，环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.10.2   |
| mxVision | 2.0.2  |
| python   | 3.7.5  |

确保环境中正确安装mxVision SDK。

在编译运行项目前，需要设置环境变量：

```
export MX_SDK_HOME=${SDK安装路径}/mxVision
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=${MX_SDK_HOME}/python:${PYTHONPATH}

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

- 环境变量介绍

```
MX_SDK_HOME：MindX SDK mxVision的根安装路径，用于包含MindX SDK提供的所有库和头文件。  
LD_LIBRARY_PATH：提供了MindX SDK已开发的插件和相关的库信息。  
install_path：ascend-toolkit的安装路径。  
PATH：添加python的执行路径和atc转换工具的执行路径。  
LD_LIBRARY_PATH：添加ascend-toolkit和MindX SDK提供的库目录路径。  
ASCEND_OPP_PATH：atc转换工具需要的目录。 
```

## 3 模型获取与准备工作

**步骤1** 将所需[模型](https:///mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/FacialExpressionRecognition/model.zip)下载并解压至model文件中 

**步骤2** 将待推理的数据图片放入image文件夹中，并修改main.py中对应的图片路径


## 4 编译与运行

**步骤1** 从https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/SentimentAnalysis/data.zip 下载测试数据并解压，解压后的sample.txt和test.csv文件放在项目的data目录下。

**步骤2** 按照第 2 小节 环境依赖 中的步骤设置环境变量。

**步骤3** 按照第 3 小节 模型获取及转换 中的步骤获得 om 模型文件。

**步骤4** 在项目目录下执行命令：
python3.7 main.py
命令执行成功后在out目录下生成分类结果文件 prediction_label.txt，查看结果文件验证分类结果。在目录test下提供精度测试代码。
