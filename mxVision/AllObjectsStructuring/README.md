# 全目标结构化

## 1 简介

全目标结构化样例基于mxVision SDK进行开发，以昇腾Atlas300卡为主要的硬件平台，主要支持以下功能：

1. 目标检测：在视频流中检测出目标，本样例选用基于Yolov4-tiny的目标检测，能达到快速精准检测。
2. 动态目标识别和属性分析：能够识别出检测出的目标类别，并对其属性进行分析。
3. 人体属性分类+PersonReID：能够根据人体属性和PersonReID进行分类.
4. 人脸属性分类+FaceReID：能够根据人脸属性和FaceReID进行分类.
5. 车辆属性分类：能够对车辆的属性进行分类。


## 2 环境依赖

- 支持的硬件形态和操作系统版本

| 硬件形态                             | 操作系统版本   |
| ----------------------------------- | -------------- |
| x86_64+Atlas 300I 推理卡（型号3010） | Ubuntu 18.04.1 |
| x86_64+Atlas 300I 推理卡 （型号3010）| CentOS 7.6     |
| ARM+Atlas 300I 推理卡 （型号3000）   | Ubuntu 18.04.1 |
| ARM+Atlas 300I 推理卡 （型号3000）   | CentOS 7.6     |

- 软件依赖

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.5.1+ |
| mxVision | 0.2    |
| Python   | 3.7.5  |



## 3 代码主要目录介绍

本代码仓名称为mxSdkReferenceApps，工程目录如下图所示：

```
├── mxVision
│   ├── AllObjectsStructuring
│   |   ├── pipeline
│   |   │   └── AllObjectsStructuring.pipeline
│   |   ├── plugins
│   |   │   ├── MpObjectSelection
|   |   |   |   ├── CMakeLists.txt
|   |   |   |   ├── MpObjectSelection.cpp
|   |   |   |   └── MpObjectSelection.h
│   |   │   └── MxpiFaceSelection
│   |   |       ├── CMakeLists.txt
│   |   │       ├── MxpiFaceSelection.cpp
│   |   │       └── MxpiFaceSelection.h
│   |   ├── models
│   |   ├── CMakeLists.txt
│   |   ├── README.zh.md
│   |   ├── build.sh
│   |   ├── main.py
│   |   └── run.sh
```



## 4 准备

**步骤1：** 参考安装教程《mxVision 用户指南》安装 mxVision SDK。

**步骤2：** 配置 mxVision SDK 环境变量。

`export MX_SDK_HOME=${安装路径}/mxVision `

注：本例中mxVision SDK安装路径为 /root/MindX_SDK。

**步骤3：** 在项目根目录下 AllObjectStructuring/ 创建目录models `mkdir models` ，联系我们获取最新模型，并放到项目根目录下 AllObjectStructuring/models/ 目录下。

**步骤4：** 在项目根目录下 AllObjectStructuring/ 创建目录faces_to_register `mkdir faces_to_register` ，将用来注册入库的人脸照片放到项目根目录下 AllObjectStructuring/faces_to_register/ 目录下。faces_to_register目录中可以存放子文件夹，照片格式必须为.jpg，且子文件夹名称必须为英文字符。如果不需要接入特征检索功能，此步骤可忽略。

**步骤5：** 修改项目根目录下 AllObjectStructuring/pipeline/AllObjectsStructuring.pipeline文件：

①：将所有“rtspUrl”字段值替换为可用的 rtsp 流源地址（需要自行准备可用的视频流，目前只支持264格式的rtsp流，264视频的分辨率范围最小为128 * 128，最大为4096 * 4096，不支持本地视频），配置参考如下：
```bash
rstp流格式为rtsp://${ip_addres}:${port}/${h264_file}
例：rtsp://xxx.xxx.xxx.xxx:xxxx/xxxx.264
```

②：将所有“deviceId”字段值替换为实际使用的device的id值，可用的 device id 值可以使用如下命令查看：

`npu-smi info`

**步骤6：** 修改项目根目录下 AllObjectStructuring/pipeline/face_registry.pipeline文件：

①：将所有“deviceId”字段值替换为实际使用的device的id值，勿与AllObjectStructuring.pipeline使用同一个deviceId。可用的 device id 值可以使用如下命令查看：

`npu-smi info`

**步骤7：** 编译mxSdkReferenceApps库中的插件：

在当前目录下，执行如下命令：

`bash build.sh`

**步骤8：** 在当前目录下，安装必要python库：

`pip3.7.5 install -r requirements.txt`

**步骤9：** 请在昇腾社区下载特征检索源码包https://www.hiascend.com/software/mindx-sdk/mxindex，并根据readme来搭建特征检索库。如果不需要接入特征检索功能，此步骤可忽略。

注：当前版本特征检索库缺少本例中人脸检索所需部分算子（Flat，IVFSQ8），需自行生成，请参考特征检索readme 4.2.2：

首先进入特征检索 src/ascendfaiss/tools 目录，

在该目录下执行生成算子命令，当前特征检索版本算子生成命令如下：

`python flat_min64_generate_model.py -d 256`

`python ivfsq8_generate_model.py -d 256 -c 8192`

生成算子后将算子模型文件移动到上级目录的modelpath目录下：

`mv op_models/* ../modelpath`

重新执行环境部署：

`bash install.sh <driver-untar-path>`

`<driver-untar-path>`为“`Ascend310-driver-{software version}-minios.aarch64-src.tar.gz`”文件解压后的目录，例如文件在“`/usr/local/software/`”目录解压，则`<driver-untar-path>`为“`/usr/local/software/`” 。本步命令用于实现将device侧检索daemon进程文件分发到多个device，执行命令后Ascend Driver中device的文件系统会被修改，所以需要执行**“`reboot`”**命令生效。

准确的算子生成方式需以特征检索 readme 4.2.2 为准。



## 5 运行

### 不带检索

运行
`bash run.sh`

正常启动后，控制台会输出检测到各类目标的对应信息。


### 带检索
需要在项目根目录下 AllObjectsStructuring/util/arguments.py 配置检索大小库运行的芯片id

配置检索小库运行芯片id，根据实际情况修改`default`值，勿与注册人脸、全目标结构化pipeline使用同一芯片
```bash
    parser.add_argument('-index-little-device-ids',
                        type=int,
                        nargs="+",
                        default=[2],
                        help='specify the device assignment for little index.',
                        dest='index_little_device_ids')
```
配置检索大库运行芯片id，根据实际情况修改`default`值，勿与注册人脸、全目标结构化pipeline使用同一芯片
```bash
    parser.add_argument('-index-large-device-ids',
                        type=int,
                        nargs="+",
                        default=[3],
                        help='specify the device assignment for large index.',
                        dest='index_large_device_ids')
```

运行
`bash run.sh index`

正常启动后，控制台会输出检测到人脸目标的对应索引信息。



## 6 参考链接

MindX SDK社区链接：https://www.hiascend.com/software/mindx-sdk



## 7 FAQ

### 7.1 运行程序时,LibGL.so.1缺失导致导入cv2报错 

**问题描述：**
运行程序时报错："ImportError: libGL.so.1: cannot open shared object file: No such file or directory"

**解决方案：**

如果服务器系统是Debian系列，如Ubantu，执行下列语句：
```bash
sudo apt update
sudo apt install libgl1-mesa-glx
```

如果服务器系统是RedHat系列，如Centos，执行下列语句：
```bash
yum install mesa-libGL
```