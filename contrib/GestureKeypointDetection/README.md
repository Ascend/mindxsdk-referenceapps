# 手势关键点检测
## 1. 介绍
手势关键点检测插件基于 MindXSDK 开发，在晟腾芯片上进行人手检测以及手势关键点检测，将检测结果可视化并保存。输入一幅图像，可以检测得到图像中所有检测到的人手以及手势关键点连接成手势骨架。

人手识别是在输入图像上对人手进行检测，采取yolov3模型，将待检测图片输入模型进行推理，推理得到所有人手的位置坐标，之后使用方框渲染标明在原来的图片上。本方案可以对人手正面、侧面、背面，多手进行检测。但对于手部被遮挡的情况，不能准确检测出来。
手势关键点检测是指在人手识别的基础上对人手识别推理出的结果进行手势21个关键点进行检测，包括大拇指，食指，中指，无名指上各有4个关键点，还有手腕上的一个关键点。然后将关键点正确配对组成相应的手势骨架，展示手势姿态。本方案采取resnet50模型，将待检测图片输入模型进行推理，推理得到包含手势关键点信息和关键点之间关联度信息的两个特征图，首先从关键点特征图中提取得到所有候选手势关键点，然后结合关联度信息特征图将不同的关键点组合连接成为手势骨架，再将所有手势骨架连接组成不同的手势，最后将关键点和手势信息标注在输入图像上，描绘手势形态。


### 1.1 支持的产品

本项目以昇腾Atlas310卡为主要的硬件平台。



### 1.2 支持的版本

支持的SDK版本为2.0.4
支持的cann版本为5.0.4


### 1.3 软件方案介绍

基于MindX SDK的人体关键点检测业务流程为：待检测图片通过 appsrc 插件输入，然后使用图像解码插件mxpi_imagedecoder对图片进行解码，再通过图像缩放插件mxpi_imageresize将图像缩放至满足检测模型要求的输入图像大小要求，缩放后的图像输入模型推理插件mxpi_tensorinfer得到检测人手坐标结果，经过mxpi_objectpostprocessor后处理后把发送给mxpi_imagecrop插件作resize处理，然后发送给下一个mxpi_tensorinfer作关键点检测，从中提取关键点，最后通过输出插件 appsink 获取人体关键点。在python上实现关键点和关键点之间的连接关系，输出关键点连接形成的手势，并保存图片。本系统的各模块及功能描述如表1所示：

表1 系统方案各模块功能描述：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | 图片输入    | 获取 jpg 格式输入图片 |
| 2    | 图片解码    | 解码图片 |
| 3    | 图片缩放    | 将输入图片放缩到下一个模型指定输入的尺寸大小 |
| 4    | 模型推理    | 对输入张量进行推理，检测人手坐标 |
| 5    | 目标检测后处理推理    | 从模型推理结果进行后处理 |
| 6    | 图片缩放    | 将输入图片放缩到下一个模型指定输入的尺寸大小 |
| 7    | 模型推理    | 对输入张量进行推理，检测人手关键点 |
| 8    | 结果输出    | 将人手关键点结果输出|


### 1.4 代码目录结构与说明

本工程名称为 GestureKeypointDetection，工程目录如下所示：
```
.
├── main.py
├── detection.pipeline
├── model
│   ├── hand
│   │   ├── model_conversion.sh
│   │   ├── coco.names
│   │   ├── aipp.cfg
│   │   └── hand.cfg
│   └── keypoint
│       ├── model_conversion.sh
│       └── insert_op.cfg
└── README.md
```


## 2 环境依赖

推荐系统为ubantu 18.04，环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.5+   |
| mxVision | 2.0.4  |
| python   | 3.9.2  |

确保环境中正确安装mxVision SDK。

在编译运行项目前，需要设置环境变量：
```
export MX_SDK_HOME=${SDK安装路径}/mxVision
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub:${LD_LIBRARY_PATH}"
export PYTHONPATH="${MX_SDK_HOME}/python:${PYTHONPATH}"
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins"
```

- 环境变量介绍

```
MX_SDK_HOME: mxVision SDK 安装路径
LD_LIBRARY_PATH: lib库路径
PYTHONPATH: python环境路径
```


## 3. 模型转换
### 3.1 yolov3模型转换
本项目中适用的第一个模型是 yolov3 模型，参考实现代码：https://codechina.csdn.net/EricLee/yolo_v3， 选用的模型是该 pytorch 项目中提供的模型。
下载onnx模型，下载链接为：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/GestureKeypointDetection/yolov3_hand.onnx
使用模型转换工具 ATC 将 onnx 模型转换为 om 模型，模型转换工具相关介绍参考链接：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html
自行转换模型步骤如下：
1. 从上述 onnx 模型下载链接中下载 onnx 模型至 ``model/hand`` 文件夹下，文件名为：yolov3_hand.onnx 。
2. 进入 ``model/hand`` 文件夹下执行命令：
```
bash model_convertion.sh
```
执行该命令后会在当前文件夹下生成项目需要的模型文件 hand.om。执行后终端输出为：
```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```
表示命令执行成功。

### 3.2 resnet50模型转换

第二个模型是 resnet50 模型，参考实现代码：https://codechina.csdn.net/EricLee/handpose_x。
下载onnx模型，下载链接为https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/GestureKeypointDetection/resnet_50_size-256.onnx ，使用模型转换工具 ATC 将 onnx 模型转换为 om 模型 。

自行转换模型步骤如下：
1. 从上述 onnx 模型下载链接中下载 onnx 模型至 ``model/hand_keypoint`` 文件夹下，文件名为：resnet_50_size-256.onnx 。
2. 进入 ``model/hand`` 文件夹下执行命令：
```
bash model_convertion.sh
```
执行该命令后会在当前文件夹下生成项目需要的模型文件 hand_keypoint.om。执行后终端输出为：
```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```

## 4. 运行
**步骤1** 根据环境SDK的安装路径配置detection.pipeline中的{$MX_SDK_HOME}。

**步骤2** 按照第 2 小节 **环境依赖** 中的步骤设置环境变量。

**步骤3** 按照第 3 小节 **模型转换** 中的步骤获得 om 模型文件，放置在 ``model/hand`` 和 ``model/hand_keypoint`` 目录下。

**步骤4** 网上下载手势图片。

**步骤5** 图片检测。将关于人手手势的图片放在项目目录下，命名为 test.jpg。在该图片上进行检测，执行命令：
```
python3 main.py test.jpg
```
命令执行成功后在当前目录下生成检测结果文件 result_test.jpg，查看结果文件验证检测结果。

