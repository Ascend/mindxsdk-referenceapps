# 全目标结构化V2

## 1 介绍

全目标结构化V2样例基于mxVision SDK进行开发，以310P为主要的硬件平台，主要支持以下功能：

1. 目标检测：在视频流中检测出目标，本样例选用基于Yolov4-tiny的目标检测，能达到快速精准检测。
2. 动态目标识别和属性分析：能够识别出检测出的目标类别，并对其属性进行分析。
3. 行人属性分类+PersonReID：能够根据人体属性和PersonReID进行分类。（行人属性，行人特征）
4. 人脸属性分类+FaceReID：能够根据目标属性和FaceReID进行分类。(人脸属性，人脸特征)
5. 车辆属性分类：能够对车辆的属性进行分类。(车辆属性，车辆特征以及车牌检测/识别)

### 1.1 支持的产品
310P: Atlas 300I pro Atlas 300V pro

### 1.2 支持的版本
eg：推荐系统为ubuntu 18.04或centos 7.6，环境依赖软件和版本如下表：

  | 软件名称 | 版本 |
  | -------- | -----|
  | cmake    | 3.10.2|
  | mxVision | 5.0RC2|
  | gcc      | 7.5.0 |
  | ffmpeg   | 4.2.1 |

### 1.3 代码目录结构与说明
```
├── AllObjectsStructuring
|   ├── include
|   │   ├── taskflow
|   ├── models
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

## 2 模型下载与转换

**步骤1：** 在项目根目录下 AllObjectStructuring2/ 创建目录models `mkdir models` ，获取[模型](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/mxVision/AllObjectsStructuringV2/AllObjectsStructuringV2_models.zip)，并放到项目根目录下 AllObjectStructuring2/models/ 目录下。

**步骤2：** 执行以下命令使用atc命令进行模型转换
```bash
atc --model=./yolov4_improve/yolov4-tiny-customized.pb --framework=3 --input_shape="inputs:1,416,416,3" -output=./yolov4_improve/yolov4_detection_1batch --insert_op_conf=./yolov4_improve/aipp_yolov4.cfg --soc_version=Ascend310P3

atc --model=./pedestrianattribute/pedestrian_attribute.prototxt --weight=./pedestrianattribute/pedestrian_attribute.caffemodel --framework=0 -output=./pedestrianattribute/pede_attr --insert_op_conf=./pedestrianattribute/aipp.cfg --soc_version=Ascend310P3

atc --model=./car_plate_recognition/car_plate_recognition.prototxt --weight=./car_plate_recognition/car_plate_recognition.caffemodel --framework=0 -output=./car_plate_recognition/car_plate_recognition --insert_op_conf=./car_plate_recognition/aipp.cfg --soc_version=Ascend310P3

atc --model=./car_plate_detection/car_plate_detection.prototxt --weight=./car_plate_detection/car_plate_detection.caffemodel --framework=0 -output=./car_plate_detection/car_plate_detection --insert_op_conf=./car_plate_detection/aipp.cfg --soc_version=Ascend310P3

atc --model=./pedereid/pedestrian_reid.prototxt --weight=./pedereid/pedestrian_reid.caffemodel --framework=0 -output=./pedereid/pede_reid --insert_op_conf=./pedereid/aipp.cfg --soc_version=Ascend310P3

atc --model=./facequality/face_quality_batch_1.prototxt --weight=./facequality/face_quality.caffemodel --framework=0 -output=./facequality/face_quality_improve --insert_op_conf=./facequality/aipp.cfg --soc_version=Ascend310P3

atc --model=./faceattr/face_attribute_batch_1.prototxt --weight=./faceattr/face_attribute.caffemodel --framework=0 -output=./faceattr/face_attribute_batch_1 --insert_op_conf=./faceattr/aipp.cfg --soc_version=Ascend310P3

atc --model=./facefeature/face_feature_batch_1.prototxt --weight=./facefeature/face_feature.caffemodel --framework=0 -output=./facefeature/face_feature_batch_1 --insert_op_conf=./facefeature/aipp.cfg --soc_version=Ascend310P3

atc --model=./motorattr/vehicle_attribute.pb --framework=3 -output=./motorattr/vehicle_attribute --insert_op_conf=./motorattr/aipp.cfg --soc_version=Ascend310P3
```

## 3 安装依赖
**步骤1：** 安装ffmpeg，源码下载地址： https://github.com/FFmpeg/FFmpeg/archive/n4.2.1.tar.gz
```shell
./configure --prefix=/usr/local/ffmpeg --enable-shared
make -j8
make install
```

**步骤2：** 安装[taskflow](https://github.com/taskflow/taskflow/releases), 在根目录下创建include目录，从github上下载taskflow的源码,解压后将taskflow目录复制到include目录中。

## 4 修改模型路径和测试视频

**步骤1：** 在main.cpp中修改生成的对应模型以及配置路径。如yoloModelPath，yoloConfigPath，yoloLabelPath等。
```shell
std::string yoloModelPath = "../models/yolov4_improve/yolov4_detection_1batch.om";
std::string yoloConfigPath = "../models/yolov4_improve/yolov4.cfg";
std::string yoloLabelPath = "../models/yolov4_improve/coco.names";

std::string vehicleAttrModelPath = "../models/motorattr/vehicle_attribute.om";
std::string vehicleAttrConfigPath = "../models/motorattr/vehicle_attribute.cfg";
std::string vehicleAttrLabelPath = "../models/motorattr/vehicle_attribute_coco.names";

std::string carPlateDetectModelPath = "../models/car_plate_detection/car_plate_detection.om";
std::string carPlateDetectConfigPath = "../models/car_plate_detection/car_plate_detection.cfg";
std::string carPlateDetectLabelPath = "../models/car_plate_detection/car_plate_detection.names";
std::string carPlateRecModelPath = "../models/car_plate_recognition/car_plate_recognition.om";

std::string pedestrianAttrModelPath = "../models/pedestrianattribute/pede_attr.om";
std::string pedestrianAttrConfigPath = "../models/pedestrianattribute/pedes_attr.cfg";
std::string pedestrianAttrLabelPath = "../models/pedestrianattribute/coco.names";
std::string pedestrianFeatureModelPath = "../models/pedereid/pede_reid.om";

std::string faceLandmarkModelPath = "../models/facequality/face_quality_improve.om";
std::string faceAttributeModelPath = "../models/faceattr/face_attribute_batch_1.om";
std::string faceAttributeConfigPath = "../models/faceattr/yolov3-tiny-addpad.cfg";
std::string faceAttributeLabelPath = "../models/faceattr/coco.names";
std::string faceFeatureModelPath = "../models/facefeature/face_feature_batch_1.om";

```

**步骤2：** 在main.cpp中修改测试视频路径，当前使用本地的文件作为视频流输入。
```
std::fill(filePaths.begin(), filePaths.end(), "../test.264"); // 如需通过rtsp获取视频流，请将其注释。
```


**步骤3：** 在main.cpp中修改视频路数以及并行度。
+ numChannel 视频路数
+ numWorker woker数量
+ numLines 并行度，数值为numChannel/numWorker:单个worker能够处理视频的路数。
```
const size_t numChannel = 70;
const size_t numWorker = 7;
const size_t numLines = 10;
# 说明：通过查询npu-smi info，查询内存和AICore用量，根据实际情况调整worker和并行度
```

## 5 编译与运行
**步骤1：** 配置环境变量。
```
. ${sdk_path}/set_env.sh
. ${ascend_toolkit_path}/set_env.sh
export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH
```

**步骤2**：在项目根目录下创建build文件夹，使用cmake命令进行编译，生成可执行文件：

```bash
## 创建build目录
mkdir build
cd build
## 使用cmake命令进行编译
cmake ..
make -j
## 编译运行
./main
```

## 6 通过网络拉流来获取视频输入
**步骤1：** 将本地获取视频流的代码注释掉。

```
std::this_thread::sleep_for(std::chrono::milliseconds(FPSSleep30)); // 手动控制帧率，如需通过rtsp获取视频流，请将其注释
std::fill(filePaths.begin(), filePaths.end(), "../test.264"); // 如需通过rtsp获取视频流，请将其注释。

```
**步骤2：** 根据下面的文档，[创建网络视频流](https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/reference_material/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md)


**步骤3：** 在main.cpp中添加拉流代码
```
    std::vector<std::string> filePaths(numChannel);
    std::fill(filePaths.begin(), filePaths.end(), "../test.264"); // 如需通过rtsp获取视频流，请将其注释。
    AVFormatContext *pFormatCtx[numChannel];
    AVPacket pkt[numChannel];
    std::vector<uint32_t> frameIDs(numChannel, deviceID);
    tf::Executor executor(EXECUTOR_NUM);
    tf::Taskflow taskflow;
...

```
将std::fill这一行代码注释后，在下面添加以下代码，用于网络拉流。
```
for (size_t i = 0; i < numChannel; i++) {
    int port = ${port} + i;
    filePaths[i] = "rtsp://${ip_address}:" + std::to_string(port) + "/test.264"
}

```
port:起始端口
nums:端口数量
ip_address:ip地址
如果是不同名称的264视频，则手动批量添加、
```
std::vector<std::string> filePaths(numChannel);
filepaths[0] = "rtsp://xxx.xxx.xxx.xxx:xxxx/xxx.264"
filepaths[1] = "rtsp://xxx.xxx.xxx.xxx:xxxx/xxx.264"
...

```

**步骤4**：编译运行

```bash
## 创建build目录
mkdir build
cd build
## 使用cmake命令进行编译
cmake ..
make -j
## 编译运行
./main
```

## 6 常见问题

请按照问题重要程度，详细列出可能要到的问题，和解决方法。

### 3.1 模型路径错误问题

**问题描述：**

Failed to get model, the model path is invalidate.

**解决方案：**
可执行文件main和模型相对路径不匹配，确认可执行文件和model目录相对路径，或者在main.cpp中直接使用模型全路径。