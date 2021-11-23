# C++ 基于MxBase的车流量统计开发
## 1 介绍
车流量统计是指对视频中的车辆进行计数，实现对本地视频（H264）进行车辆追踪并计数，最后生成可视化结果。车流统计分为五个步骤：车辆视频流读取、车辆检测、车辆追踪、车辆计数以及结果可视化。
### 1.1 支持的产品
支持昇腾310芯片
### 1.2 支持的版本
本样例配套的CANN版本为3.3.0，MindX SDK版本为2.0.2
### 1.3 软件方案介绍
车流统计项目实现：输入类型是视频数据（需要将视频转换为.264的视频格式），ffmpeg打开视频流获取视频帧信息，图像经过尺寸大小变换，满足模型的输入尺寸要求；将尺寸变换后的图像数据依次输入Yolov4检测模型进行推理，模型输出经过后处理后，使用SORT算法进行车辆追踪得到车辆轨迹，再设置标志对车辆进行计数，最后得到某时刻已经通过的车辆数。

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
│   ├── aipp_yolov3_416_416.aippconfig
│   ├── coco.names
│   ├── yolov3_tf_bs1_fp16.om
│   ├── yolov4_bs.om
├── BlockingQueue
│   ├── BlockingQueue.h
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
├── README.md
└── run.sh
```



### 1.5 技术实现流程图

![Image text](https://gitee.com/wu-jindge/mindxsdk-referenceapps/raw/master/contrib/VehicleCounting/img/process.JPG)

## 2 环境依赖
环境依赖软件和版本如下表：



| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 2.0.2        | mxVision软件包                | [链接](https://www.hiascend.com/software/mindx-sdk/mxvision) |
| Ascend-CANN-toolkit | 3.3.0        | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | Ubuntu 18.04 | 操作系统                      | Ubuntu官网获取                                               |
| ffmpeg             | 4.2.1        | 视频转码解码组件              | [安装教程](https://bbs.huaweicloud.com/forum/thread-142431-1-1.html)|                                               
| pc端ffmpeg         | 2021-09-01   | 将视频文件格式转换为.264      | [安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/pc%E7%AB%AFffmpeg%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B.md)|



## 3 模型转换

**步骤1** 模型获取
在ModelZoo上下载[YOLOv4模型](https://www.hiascend.com/zh/software/modelzoo/detail/1/abb7e641964c459398173248aa5353bc)（或者[YOLOv3模型](https://www.hiascend.com/zh/software/modelzoo/detail/C/210261e64adc42d2b3d84c447844e4c7)，选择“历史版本”中版本1.1下载）

**步骤2** 模型存放
将获取到的YOLOv4模型onnx文件存放至："样例项目所在目录/model/"。

**步骤3** 模型转换
在onnx文件所在目录下执行一下命令

```
# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 执行，转换YOLOv4/YOLOv3模型
# Execute, transform YOLOv4/YOLOv3 model.

YOLOv4:
atc --model=./yolov4_dynamic_bs.onnx --framework=5 --output=yolov4_bs --input_format=NCHW --soc_version=Ascend310 --insert_op_conf=./aipp_yolov4_608_608.config --input_shape="input:1,3,608,608" --out_nodes="Conv_434:0;Conv_418:0;Conv_402:0"
YOLOv3:
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.config --input_shape="input/input_data:1,416,416,3" --out_nodes="conv_lbbox/BiasAdd:0;conv_mbbox/BiasAdd:0;conv_sbbox/BiasAdd:0"
# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
```

执行完模型转换脚本后，会生成相应的.om模型文件。 执行完模型转换脚本后，会生成相应的.om模型文件。我们也提供了已经转换好的YOLOv4/YOLOv3 om模型：[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/VehicleCounting/modle.rar)

模型转换使用了ATC工具，如需更多信息请参考:

 https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html

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

VideoProcess.cpp文件中：
```
# 视频的宽高值
const uint32_t VIDEO_WIDTH = {视频宽度};
const uint32_t VIDEO_HEIGHT = {视频高度};

# 计数标志位的位置，(x1,y1)和(x2,y2)分别为计数标志位两个端点的像素坐标（计数标志位最好与车流方向垂直，计数效果更好）
line = {center{x1,y1}, center{x2, y2}}

# 计数参数的显示，计数参数显示在视频的左上角，如下图的样例视频中，共3个参数：total、lane_up、lane_down
  lane_up表示朝向摄像头行驶的车流统计数量，lane_down表示原理摄像头行驶的车流统计数量，total表示总共的车流统计数量
（1）若想计数单车道的车流，可在代码310-320行只保留想要进行车流统计车道的计数参数的计算和显示，例如只保留counter_up或counter_down。
（2）若想计数更多车道的车流，可增加额外类似于line={center{x1,y1},center{x2, y2}}的计数标志位和类似于counter_up和
     counter_down的计数参数以及参数的位置point={x,y}（x,y为计数参数的位置）;新的计数参数的计算需要再添加307-316行
     的代码逻辑，其中intersect函数的后两个参数为新的计数标志位的两个端点。新的计数参数和计数标志位的显示需要再添加
     317-320行的cv::line和cv::putText函数。

# 计数方法修改，如果计数标志位是南北方向，需要将第310行的：last_point[0].y>last_point[1].y
  改为last_point[0].x>last_point[1].x，因为此时车流方向为东西方向，用x轴坐标计算更为准确
```
Yolov4Detection.cpp文件中：
```
# 阈值修改，可通过修改检测置信度阈值det_threshold和非最大值抑制阈值nms_iouthreshold来调整检测效果。
```
**步骤4** 设置环境变量，
FFMPEG_HOME为ffmpeg安装的路径，MX_SDK_HOME为MindXSDK安装的路径
LD_LIBRARY_PATH 指定程序运行时依赖的动态库查找路径
```
# 执行如下命令，打开.bashrc文件
vi .bashrc
# 在.bashrc文件中添加以下环境变量
MX_SDK_HOME=${SDK安装路径}
FFMPEG_HOME=${FFMPEG安装路径} 
# 若环境中没有安装ffmpeg，请联系支撑人员

LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:/usr/local/python3.7.5/lib:${FFMPEG_HOME}/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH}
# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

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
chmod +x run.sh
bash run.sh
```
**步骤7** 查看结果

执行run.sh完毕后，图片可视化结果会被保存在工程目录下result文件夹中，视频可视化结果会被保存在工程目录下result1文件夹中

样例视频来源：[链接](https://github.com/jjw-DL/YOLOV3-SORT/tree/master/input)
![Image text](https://gitee.com/wu-jindge/mindxsdk-referenceapps/raw/master/contrib/VehicleCounting/img/result.png)

## 5 常见问题
### 模型更换问题
**问题描述** 在用YOLOv4模型替换YOLOv3模型的时候，由于模型的输入的图片resize大小不一样以及模型输出的通道顺序也不一样，会导致模型无法正常推理和后处理

**解决方案** 将YolovDetection.cpp中的图片resize大小由416x416改为608x608，并将main.cpp文件里的modelType由0改为1

### 更换视频样例问题
在测试更换视频样例时需要在VideoProcess.cpp文件中，设置视频的宽高值为样例的实际宽高值。并且车辆计数所用的标志位要根据视频实际情况和自己所想要计数的位置来重新规划
