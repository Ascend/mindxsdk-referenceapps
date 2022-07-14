# 人体关键点检测
## 1. 介绍
人体关键点检测插件基于 MindXSDK 开发，在晟腾芯片上进行人体检测以及人体势关键点检测，将检测结果可视化并保存。输入一幅图像，可以检测得到图像中所有检测到的人以及人体势关键点连接成人体骨架。

人体识别是在输入图像上对人体进行检测，采取yolov3模型，将待检测图片输入模型进行推理，推理得到所有人体的位置坐标，之后根据人体的数量，使用方框在原来的图片上分别进行剪裁。人体关键点检测插件基于 

人体关键点检测是指在输入图像上对指定的 17 类人体骨骼关键点位置进行检测，包括左腿、右腿、脑袋等。然后将关键点正确配对组成相应的人体骨架，展示人体姿态，共 19 类人体骨架，如左肩和左肘两个关键点连接组成左上臂，右膝和右踝两个关键点连接组成右小腿等。本方案采取YOLOv3与3DMPPE-ROOTNET模型，将待检测图片输入模型进行推理，推理得到包含人体关键点的根节点信息，以及预先提供的其他16类关键点，然后结合人体信息特征将不同的关键点组合连接成为人体骨架，再将所有人体骨架连接组成不同的人体，最后将关键点和骨架信息标注在输入图像上，描绘人体姿态。本方案可以对遮挡人体、小人体、密集分布人体等进行检测，还适用于不同姿态（蹲坐、俯卧等）、不同方向（正面、侧面、背面等）以及模糊人体关键点检测等多种复杂场景。


### 1.1 支持的产品

本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.2 支持的版本

支持的SDK版本为2.0.4
支持的cann版本为5.0.4


### 1.3 软件方案介绍

基于MindX SDK的人体关键点检测业务流程为：待检测图片通过 appsrc 插件输入，然后使用图像解码插件mxpi_imagedecoder对图片进行解码，再通过图像缩放插件mxpi_imageresize将图像缩放至满足检测模型要求的输入图像大小要求，缩放后的图像输入模型推理插件mxpi_tensorinfer得到检测人体坐标结果，经过mxpi_objectpostprocessor后处理后把发送给mxpi_imagecrop插件作resize处理，然后再计算一个参数，将mxpi_imagecrop插件的结果和参数一起发送给下一个mxpi_tensorinfer作关键点检测，从中提取关键点，最后通过输出插件 appsink 获取人体关键根节点。在python上实现关键点和关键点之间的连接关系，输出关键点连接形成的人体关键点，并保存图片。本系统的各模块及功能描述如表1所示：

表1 系统方案各模块功能描述：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | 图片输入    | 获取 jpg 格式输入图片 |
| 2    | 图片解码    | 解码图片 |
| 3    | 图片缩放    | 将输入图片放缩到下一个模型指定输入的尺寸大小 |
| 4    | 模型推理    | 对输入张量进行推理，检测人体坐标 |
| 5    | 目标检测后处理推理    | 从模型推理结果进行后处理 |
| 6    | 图片缩放    | 将输入图片放缩到下一个模型指定输入的尺寸大小 |
| 7    | 模型推理    | 对输入张量进行推理，检测人体关键点 |
| 8    | 结果输出    | 将人体关键点结果输出 |


### 1.4 代码目录结构与说明

本工程名称为 PeopleKeypointDetection，工程目录如下所示：
```
.
├── main.py
├── pipeline
│   ├── detection_3d.pipeline
│   └── detection_yolov3.pipeline
├── model
│   ├── people
│   │   ├── model_conversion.sh
│   │   ├── coco.names
│   │   ├── yolov3_tf_aipp.cfg
│   │   └── people.cfg
│   └── keypoint
│       └── model_conversion.sh
├── eval
│   ├── eval.py
│   └── MuPoTS_gt.json
├── item.ini
├── test.jpg
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
export LD_LIBRARY_PATH=${MX_SDK_HOME}/python:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/include:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/5.0.4/acllib/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl:${MX_SDK_HOME}/python:/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl:${MX_SDK_HOME}/python/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
```

- 环境变量介绍

```
MX_SDK_HOME: mxVision SDK 安装路径
LD_LIBRARY_PATH: lib库路径
PYTHONPATH: python环境路径
```


## 3. 模型转换
### 3.1 yolov3模型转换
本项目中适用的第一个模型是 yolov3 模型，参考实现代码：https://www.hiascend.com/zh/software/modelzoo/detail/1/ba2a4c054a094ef595da288ecbc7d7b4， 选用的模型是该 pytorch 项目中提供的模型。
下载模型，选择其中的pb模型使用。下载链接为：https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2021-12-30_tf/ATC%20YOLOv3%28FP16%29%20from%20TensorFlow%20-%20Ascend310/zh/1.1/ATC%20YOLOv3%28FP16%29%20from%20TensorFlow%20-%20Ascend310.zip
使用模型转换工具 ATC 将pb模型转换为 om 模型，模型转换工具相关介绍参考链接：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html
自行转换模型步骤如下：

1. 从上述模型下载链接中下载 ，选择pb模型至 ``model/people`` 文件夹下，文件名为：yolov3_tf.pb 。
2. 进入 ``model/people`` 文件夹下执行命令：
```
bash model_convertion.sh
```
执行该命令后会在当前文件夹下生成项目需要的模型文件 yolov3_tf_aipp.om。执行后终端输出为：
```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```
表示命令执行成功。

### 3.2 3DMPPE-ROOTNET模型转换

第二个模型是3DMPPE-ROOTNET 模型，参考实现代码：https://www.hiascend.com/zh/software/modelzoo/detail/1/c7f19abfe57146bd8ec494c0b377517c。
下载onnx模型，下载链接为https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/22.1.12/ATC%203DMPPE%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip ，使用模型转换工具 ATC 将 onnx 模型转换为 om 模型 。

自行转换模型步骤如下：
1. 从上述 onnx 模型下载链接中下载 onnx 模型至 ``model/keypoint`` 文件夹下，文件名为：3DMPPE-ROOTNET.onnx 。
2. 进入 ``model/keypoint`` 文件夹下执行命令：
```
bash model_convertion.sh
```
执行该命令后会在当前文件夹下生成项目需要的模型文件3DMPPE-ROOTNET_bs1.om。执行后终端输出为：
```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```

## 4. 运行
**步骤1** 根据环境SDK的安装路径配置detection_yolov3.pipeline中的{$MX_SDK_HOME}。

**步骤2** 按照第 2 小节 **环境依赖** 中的步骤设置环境变量。

**步骤3** 按照第 3 小节 **模型转换** 中的步骤获得 om 模型文件，放置在 ``model/people`` 和 ``model/keypoint`` 目录下。

**步骤4** 使用提供的测试图片，以及对应预提供的item.ini数据。

**步骤5** 图片检测。将关于人体的图片放在项目目录下，命名为 test.jpg。在该图片上进行检测，执行命令：

```
python3 main.py
```
命令执行成功后在当前目录下生成检测结果文件 output_root_2d_x.jpg（x为0，1，2...n，n为人体的数量）、output_root_2d_pose.jpg、output_root_3d_pose.png以及bbox_root_mupots_output.json查看结果文件验证检测结果。

**步骤6** 精度验证。将bbox_root_mupots_output.json放在项目的eval目录下，使用预先提供的MuPoTS_gt.json作为金标准。进行精度验证，执行命令：

```
python3 eval.py
```

命令执行成功后在控制台查看精度结果。

