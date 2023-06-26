# C++ 基于MxBase的车流量统计开发
## 1 介绍
车流量统计是指对视频中的车辆进行计数，实现对本地视频（H264）进行车辆动向并计数，最后生成可视化结果。车流统计分为五个步骤：车辆视频流读取、车辆检测、车辆动向、车辆计数以及结果可视化。本项目实现了对单双向车道，固定摄像头的交通视频进行车流量统计。
### 1.1 支持的产品
支持Atlas 500A2
### 1.2 支持的版本
mxVision 5.0.RC1
Ascend-CANN-toolkit （310使用6.3.RC1，310B使用6.2.RC1）
### 1.3 软件方案介绍
车流统计项目实现：输入类型是视频数据（需要将视频转换为.264的视频格式），ffmpeg打开视频流获取视频帧信息，图像经过尺寸大小变换，满足模型的输入尺寸要求；将尺寸变换后的图像数据依次输入Yolov4检测模型进行推理，模型输出经过后处理后，使用SORT算法进行车辆动向得到车辆动向，再设置标志对车辆进行计数，最后得到某时刻已经通过的车辆数。

本流程的视频检测模块参考的是Ascend的参考样例：[https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/mxBaseVideoSample](http://)

表1.1 系统方案中各模块功能：
| 序号 | 子系统            | 功能描述                                                     |
| ---- | ----------------- | ------------------------------------------------------------ |
| 1    | 设备初始化        | 调用mxBase::DeviceManager接口完成推理卡设备的初始化。        |
| 2    | 视频输入          | 使用ffmpeg打开视频流获取视频帧信息                                        |
| 3    | 图像解码/图像缩放 | 调用mxBase::DvppWrapper.DvppVdec()函数完成图像解码，VpcResize()完成缩放。 |
| 4    | 模型推理/后处理   | 调用mxBase:: ModelInferenceProcessor 接口完成车辆目标检测模型推理并进行后处理  |
| 5    | 目标跟踪         | 调用mxBase::MultipleObjectTracking和Hungarian接口完成车辆目标跟踪        |
| 6    | 车辆计数         | 设置标志对通过的车辆进行计数                                |
| 7    | 保存结果         | 使用opencv进行结果可视化并保存为视频文件                      |
| 8    | 资源释放         | 调用mxBase::DeviceManager接口完成推理卡设备的去初始化。      |

### 1.4 代码目录结构与说明

本sample工程名称为VehicleCounting，工程目录如下图所示：
```
.
├── data
│   └── test1.264
├── model
│   ├── aipp_yolov4_608_608.config
│   ├── coco.names
│   ├── yolov4_bs.om
├── BlockingQueue
│   ├── BlockingQueue.h
├── ReadConfig
│   ├── GetConfig.h
│   ├── GetConfig.cpp
├── VideoProcess
│   ├── DataType.h
│   ├── Hungarian.h
│   ├── Hungarian.cpp
│   ├── KalmanTracker.h
│   ├── KalmanTracker.cpp
│   ├── MOTConnection.h
│   ├── MOTConnection.cpp
│   ├── VideoProcess.h
│   ├── VideoProcess.cpp
├── Yolov4Detection
│   ├── Yolov4Detection.h
│   ├── Yolov4Detection.cpp
├── CMakeLists.txt
├── main.cpp
├── params.config
├── README.md
└── run.sh
```



### 1.5 技术实现流程图

![Image text](https://gitee.com/wu-jindge/mindxsdk-referenceapps/raw/master/contrib/VehicleCounting/img/process.JPG)

## 2 环境依赖
环境依赖软件和版本如下表：



| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            |  5.0.RC1     | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| Ascend-CANN-toolkit | 6.3.RC1或6.2.RC1 | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | ubuntu 22.04 | 操作系统                      | Ubuntu官网获取                                               |
| ffmpeg             | 4.2.1        | 视频转码解码组件              | [安装教程](https://bbs.huaweicloud.com/forum/thread-142431-1-1.html)|                                               
| pc端ffmpeg         | 2021-09-01   | 将视频文件格式转换为.264      | [安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/pc%E7%AB%AFffmpeg%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B.md)|



## 3 模型转换

**步骤1** 模型获取
[下载模型](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/VehicleCounting/modle.rar)
                   

**步骤2** 模型存放
将获取到的YOLOv4模型onnx文件存放至："样例项目所在目录/model/"。

**步骤3** 模型转换
在onnx文件所在目录下执行一下命令

```
# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

. /usr/local/Ascend/ascend-toolkit/set_env.sh # Ascend-cann-toolkit开发套件包默认安装路径，根据实际安装路径修改
. ${MX_SDK_HOME}/mxVision/set_env.sh # ${MX_SDK_HOME}替换为用户的SDK安装路径

#执行，转换YOLOv4/YOLOv3模型
#Execute, transform YOLOv4/YOLOv3 model.

YOLOv4:
atc --model=./yolov4_dynamic_bs.onnx --framework=5 --output=yolov4_bs --input_format=NCHW --soc_version=Ascend310B1 --insert_op_conf=./aipp_yolov4_608_608.config --input_shape="input:1,3,608,608" --out_nodes="Conv_434:0;Conv_418:0;Conv_402:0"
YOLOv3:
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310B1 --insert_op_conf=./aipp_yolov3_416_416.config --input_shape="input/input_data:1,416,416,3" --out_nodes="conv_lbbox/BiasAdd:0;conv_mbbox/BiasAdd:0;convv_sbbox/BiasAdd:0"


# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
```

模型转换使用了ATC工具，如需更多信息请参考:

 https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md

备注：若推理芯片为310B，需要将atc-env脚本中模型转换atc命令中的soc_version参数设置为Ascend310B1。

## 4 编译与运行
**步骤1** 通过pc端ffmpeg软件将输入视频格式转换为.264格式，如下所示为MP4转换为h.264命令：
```
ffmpeg -i test.mp4 -vcodec h264 -bf 0 -g 25 -r 10 -s 1280*720 -an -f h264 test.264

//-bf B帧数目控制，-g 关键帧间隔控制，-s 分辨率控制 -an关闭音频， -r 指定帧率
```
**步骤2** 配置CMakeLists.txt文件中的MX_SDK_HOME与FFMPEG_PATH环境变量，将set(MX_SDK_HOME ${SDK安装路径})中的${SDK安装路径}替换为实际的SDK安装路径和set(FFMPEG_PATH ${ffmpeg安装路径})中的${ffmpeg安装路径}替换为实际的ffmpeg安装路径。

```
set(MX_SDK_HOME {SDK实际安装路径})
set(FFMPEG_PATH {ffmpeg安装路径})
```
**步骤3** 参数设置

params.config文件中：
```
# 视频的宽高值
video_width: 1280
video_height: 720
# 输出视频的帧率
frame_rate: 15
# 是否只计数单车道，1和0表示是或否
is_singlelane: 0
# 如果单车道，选择一个车道1和0表示line_down或line_up
lane_num: 2
# 计数标志位两个端点坐标
line_s_x: 0
line_s_y: 100
line_e_x: 1280
line_e_y: 100
# 计数标志位是否为垂直或接近垂直，1和0表示是或否
is_vertical: 0
# 三个计数参数的显示位置
point_x: 0
point_y: 20
point1_x: 0
point1_y: 50
point2_x: 0
point2_y: 80
# 检测置信度阈值
det_threshold: 0.55
# 非最大值抑制阈值
nms_iouthreshold: 0.6

# 计数参数显示在视频的左上角，使用样例视频双车道计数，后台共3个参数：counter_up、counter_down、counter
  视频结果中分别显示为lane_up、lane_down、total。lane_up表示朝向摄像头行驶的车流统计数量，lane_down表示原理摄像头
  行驶的车流统计数量，total表示总共的车流统计数量。若想计数单车道的车流，is_singlelane设置位1，lane_num设置位1或2
  表示只计数line_up或line_down。

# 如果计数标志位是垂直方向或接近垂直，is_vertical设置为1，因为此时车流方向为东西方向，用x坐标计算更为准确

# 阈值修改，可通过修改检测置信度阈值det_threshold和非最大值抑制阈值nms_iouthreshold来调整检测效果。
```
**步骤4** 设置环境变量，
FFMPEG_HOME为ffmpeg安装的路径
LD_LIBRARY_PATH 指定程序运行时依赖的动态库查找路径
```
export FFMPEG_HOME=${FFMPEG安装路径} 
export LD_LIBRARY_PATH=${FFMPEG_HOME}/lib:${LD_LIBRARY_PATH}

#查看环境变量
env
```
**步骤5** 编译项目文件

新建立build目录，进入build执行cmake ..（..代表包含CMakeLists.txt的源文件父目录），在build目录下生成了编译需要的Makefile和中间文件。执行make构建工程，构建成功后就会生成可执行文件。

```
mkdir build

cd build

cmake ..

make -j
Scanning dependencies of target stream_pull_test
[ 25%] Building CXX object CMakeFiles/stream_pull_test.dir/main.cpp.o
[ 50%] Building CXX object CMakeFiles/stream_pull_test.dir/VideoProcess/VideoProcess.cpp.o
[ 75%] Building CXX object CMakeFiles/stream_pull_test.dir/Yolov3Detection/Yolov4Detection.cpp.o
[100%] Linking CXX executable ../stream_pull_test
[100%] Built target stream_pull_test

# stream_pull_test就是CMakeLists文件中指定生成的可执行文件。
```

**步骤6** 运行
将**步骤1**转换的视频文件test1.264放到data/目录下，执行run.sh脚本前请先确认可执行文件stream_pull_test已生成，执行如下命令运行
```
bash run.sh
```
**步骤7** 查看结果

使用[样例视频](https://github.com/jjw-DL/YOLOV3-SORT/tree/master/input)，执行run.sh完毕后，图片可视化结果会被保存在工程目录下result文件夹中，视频可视化结果会被保存在工程目录下result1文件夹中


## 5 常见问题
### 模型更换问题
**问题描述** 在用YOLOv4模型替换YOLOv3模型的时候，由于模型的输入的图片resize大小不一样以及模型输出的通道顺序也不一样，会导致模型无法正常推理和后处理

**解决方案** 将YolovDetection.cpp中的图片resize大小由416x416改为608x608，并将main.cpp文件里的modelType由0改为1

### config文件读取异常
**问题描述** 未提及修改main.cpp中的configUtil.LoadConfiguration的路径，运行程序会报error
```
WARNING: Logging before InitGoogleLogging() is written to STDERR
E20220507 17:05:13.253134 16193 FileUtils.cpp:315] realpath parsing failed.
E20220507 17:05:13.253193 16193 ConfigUtil.cpp:247] Failed to get canonicalized file path.
W20220507 17:05:13.253223 16193 ConfigUtil.h:103] [1016][Object, file or other resource doesn't exist] Fail to read (program_name) from config, default is: mindx_sdk
W20220507 17:05:13.253239 16193 ConfigUtil.h:103] [1016][Object, file or other resource doesn't exist] Fail to read (base_filename) from config, default is: mxsdk.log
W20220507 17:05:13.253248 16193 ConfigUtil.h:103] [1016][Object, file or other resource doesn't exist] Fail to read (set_log_destination) from config, default is: 1
```

**解决方案** 将main.cpp中72行的configUtil.LoadConfiguration路径改为实际值
```
configUtil.LoadConfiguration("$ENV{MX_SDK_HOME}/config/logging.conf", configData, MxBase::ConfigMode::CONFIGFILE);
```