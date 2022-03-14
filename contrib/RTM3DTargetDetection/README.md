# 3D目标检测

## 1 介绍
三维目标检测是自动驾驶中场景感知和运动预测的重要组成部分。目前，最强大的3D探测器严重依赖于3D LIDAR激光扫描仪，因为它可以提供场景位置。然而，基于激光雷达的系统是昂贵的，不利于嵌入到当前的车辆形状。相比之下，单目摄像机设备由于价格便宜、使用方便，在许多应用场景中受到越来越多的关注。在本项目中，我们是仅从单色RGB图像进行三维目标检测。

本次使用的3D目标检测模型在图片中只能识别三类目标：'Car'，'Pedestrian'，'Cyclist'，因此本项目的使用场景一般限定于公路道路场景中。输入图片大小最好是1280*416左右，在绝大多数情况下都能将图中目标框选出来，对于'Pedestrian'，'Cyclist'检测效果不太理想，识别率不高。

### 1.1 支持的产品

昇腾310(推理)

### 1.2 支持的版本

支持的SDK版本为[2.0.2](https://www.hiascend.com/software/mindx-sdk/mxvision)。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 软件方案介绍

基于MindX SDK的人体关键点检测业务流程为：待检测图片通过 appsrc 插件输入，然后使用图像解码插件mxpi_imagedecoder对图片进行解码，再通过图像缩放插件mxpi_imageresize将图像缩放至满足检测模型要求的输入图像大小要求，缩放后的图像输入模型推理插件mxpi_tensorinfer得到检测结果，本项目使用的3D目标检测后处理插件mxpi_objectpostprocessor是继承yolov3的后处理框架开发的，用于处理推理结果，从中提取目标类别和置信度以及相对应的3D框的坐标点，采用四个ObjectInfo结构体传输一个目标的信息，最后通过输出插件 appsink 获取3D目标检测后处理插件输出结果，并在外部进行目标3D框可视化描绘。本系统的各模块及功能描述如表1所示：

表1 系统方案各模块功能描述：

| 序号 | 子系统           | 功能描述                                               |
| ---- | ---------------- | ------------------------------------------------------ |
| 1    | 图片输入         | 获取 jpg 格式输入图片                                  |
| 2    | 图片解码         | 解码图片                                               |
| 3    | 图片缩放         | 将输入图片放缩到模型指定输入的尺寸大小                 |
| 4    | 模型推理         | 对输入张量进行推理                                     |
| 5    | 3D目标检测后处理 | 从模型推理结果计算提取物体类别、置信度和坐标框信息     |
| 6    | 结果输出         | 将后处理结果输出                                       |
| 7    | 结果可视化       | 将计算提取到的目标类别以及相对应的3D框标注在相应图片上 |

### 1.4 代码目录结构与说明

本工程名称为 RTM3DTargetDetection，工程目录如下所示：

```
├── models
│   ├── coco.names                  # 标签文件，但本项目单纯调用但不使用
│   ├── dla34.aippconfig            # 模型转换aipp配置文件
│   ├── rtm3d.cfg      # 后处理配置文件，但本项目单纯调用但不使用
│   ├── model_best.onnx             # rtm3d模型onnx格式
│   └── model_best.om               # rtm3d模型om格式
├── pipeline
│   └── rtm3d.pipeline        # pipeline文件
├── plugins
│   ├── RTM3DPostProcess      # 后处理插件
│   │   ├── CMakeLists.txt
│   │   ├── build.sh
│   │   ├── RTM3DPostProcess.cpp
│   │   └── RTM3DPostProcess.h
│   └── build.sh
├── main.py
├── draw_box.py
├── build.sh # 编译头部姿态后处理插件脚本
└── test.jpg
```



### 1.5 技术实现流程图

![3D目标检测流程图](https://gitee.com/zhiwei-liao/mindxsdk-referenceapps/raw/master/contrib/RTM3DTargetDetection/images/3D%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

图1 3D目标检测流程图

![3D目标pipeline流程图](https://gitee.com/zhiwei-liao/mindxsdk-referenceapps/raw/master/contrib/RTM3DTargetDetection/images/3D%E7%9B%AE%E6%A0%87pipeline%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

图2 3D目标检测pipeline流程图

## 2 环境依赖

| 软件名称  | 版本        |
| --------- | ----------- |
| MindX SDK | 2.0.2       |
| ubantu    | 18.04.1 LTS |
| CANN      | 3.3.0       |
| Python    | 3.9.2       |

在编译运行项目前，需要设置环境变量：

- 环境变量介绍

```
export MX_SDK_HOME=${MX_SDK_HOME}  #需要在这添加自己SDK的安装路径
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
export PYTHONPATH=${MX_SDK_HOME}/python
```

## 3 模型获取和转换过程

### 3.1 模型获取

此处提供未进行模型转换的3D目标检测模型的onnx文件以及转换好的om文件：[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/RTM3DTargetDetection/model.zip)

此处提供项目的数据集（ left color images of object data set (12 GB)），其压缩包中包含testing和training两个数据集：[下载地址](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

注：下载后请将模型放置于models目录下

### 3.2 onnx转om

在models文件目录下，需要执行的模型转换脚本示例如下：

```
# 该脚本用来将.onnx模型文件转换成.om模型文件

# 设置环境变量（请确认install_path路径是否正确）
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 执行，转换rtm3d模型
atc --model=./model_best.onnx --framework=5 --output=./model_best --soc_version=Ascend310 --input_format=NCHW --input_shape="images:1,3,416,1280" --output_type=FP32 --insert_op_conf=./dla34.aippconfig
```

执行该命令后会在当前文件夹下生成项目需要的模型文件 model_best.om。执行后终端输出为：

```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```



## 4 编译与运行
示例步骤如下：

**步骤1** 

在项目目录下执行。

```
bash build.sh
```

执行后会在该项目目录下的plugins文件中生成一个含有librtm3dpostprocess.so的plugin文件，目录排列即：

```
RTM3DTargetDetection/plugins/plugin/librtm3dpostprocess.so
```

**步骤2** 

在测试集目录下选择一张png文件，重命名（非中文名）并更改格式为test.jpg，放入项目根目录中，再执行：

```
python3 main.py --input-image test.jpg
```