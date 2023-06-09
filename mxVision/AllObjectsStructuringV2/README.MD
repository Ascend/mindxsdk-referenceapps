# 全目标结构化V2

## 1 介绍

全目标结构化V2样例基于mxVision SDK进行开发，以XXXXX为主要的硬件平台，主要支持以下功能：

1. 目标检测：在视频流中检测出目标，本样例选用基于Yolov4-tiny的目标检测，能达到快速精准检测。
2. 动态目标识别和属性分析：能够识别出检测出的目标类别，并对其属性进行分析。
3. 人体属性分类+PersonReID：能够根据人体属性和PersonReID进行分类.
4. 目标属性分类+FaceReID：能够根据目标属性和FaceReID进行分类.
5. 车辆属性分类：能够对车辆的属性进行分类。

### 1.1 支持的产品
(硬件)

### 1.2 支持的版本
(软件 包括第三方依赖)
请列出环境依赖软件和版本。

eg：推荐系统为ubuntu 18.04或centos 7.6，环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| -------- | ------ |
| XXX软件  | 版本号 |
|          |        |
|          |        |


#### 依赖安装
（依赖搭建安装或者获取方式的具体步骤）

### 1.3 代码目录结构与说明
```
├── AllObjectsStructuring
|   ├── postprocessor
|   │   ├── carPlateDetectionPostProcess // 车牌检测
|   |   |   ├── CarPlateDetectionPostProcess.cpp
|   |   |   ├── CarPlateDetectionPostProcess.h
|   |   |   └── CMakeLists.txt
|   |   ├── carPlateRecognitionPostProcess // 车牌识别
|   |   |   ├── CarPlateRecognitionPostProcess.cpp
|   |   |   ├── CarPlateRecognitionPostProcess.h
|   |   |   └── CMakeLists.txt
|   |   ├── faceAlignment // 人脸对齐
|   |   |   ├── FaceAlignment.cpp
|   |   |   ├── FaceAlignment.h
|   |   |   └── CMakeLists.txt
|   |   ├── faceLandmark // 人脸关键点
|   |   |   ├── FaceLandmark.cpp
|   |   |   ├── FaceLandmark.h
|   |   |   └── CMakeLists.txt
|   |   ├── resnetAttributePostProcess // 属性识别
|   |   |   ├── ResnetAttributePostProcess.cpp
|   |   |   ├── ResnetAttributePostProcess.h
|   |   |   └── CMakeLists.txt
|   |   ├── resnetFeaturePostProcess // 特征识别
|   |   |   ├── ResnetFeaturePostProcess.cpp
|   |   |   ├── ResnetFeaturePostProcess.h
|   |   |   └── CMakeLists.txt
|   |   └── CMakeLists.txt
|   └── utils
|       ├── multiObjectTracking
|       |   ├── multiObjectTracking.cpp
|       |   ├── multiObjectTracking.h
|       |   └── CMakeLists.txt
|       ├── objectSelection
|       |   ├── objectSelection.cpp
|       |   ├── objectSelection.h
|       |   └── CMakeLists.txt
|       └── CMakeLists.txt
├── main.cpp
├── CMakeLists.txt
└── README.zh.md
```


## 2 编译与运行

在编译运行项目前，需要设置环境变量：

（请列出所需环境变量，作用，若不涉及个人变量，列出实际设置环境变量的命令）

- 环境变量介绍

```
列出具体执行命令（简化）
. ${sdk_path}/set_env.sh
. ${ascend_toolkit_path}/set_env.sh
```

（描述项目安装运行的全部步骤，，如果不涉及个人路径，请直接列出具体执行命令）

示例步骤如下：



## 3 常见问题

请按照问题重要程度，详细列出可能要到的问题，和解决方法。

### 3.1 XXX问题

**问题描述：**

截图或报错信息

**解决方案：**

详细描述解决方法。