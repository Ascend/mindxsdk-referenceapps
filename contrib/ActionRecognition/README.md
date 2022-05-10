# ActionRecgnition

## 1 介绍

本开发样例演示动作识别系统 ActionRecgnition，供用户参考。
本系统基于mxVision SDK进行开发，以昇腾Atlas300卡为主要的硬件平台，主要应用于单人独处、逗留超时、快速移动、剧烈运动、离床检测、攀高检测六种应用场景。

1. 单人独处：识别出单个人独处场景后报警。
2. 逗留超时：识别出单人或多人在区域内长时间逗留的情况并发出警报。
3. 快速移动：检测出视频中单人或多人行进速度大于阈值的情况，并发出警报。
4. 剧烈运动：检测到视频流中有剧烈运动并进行报警。
5. 离床检测：检测出视频中行人离开指定区域的情况并报警。
6. 攀高检测：检测出行人中心点向上移动的情况，并发出警报。

## 2 环境依赖

* 支持的硬件形态和操作系统版本

  | 硬件形态                              | 操作系统版本   |
  | ------------------------------------- | -------------- |
  | x86_64+Atlas 300I 推理卡（型号3010）  | Ubuntu 18.04.1 |
  | x86_64+Atlas 300I 推理卡 （型号3010） | CentOS 7.6     |
  | ARM+Atlas 300I 推理卡 （型号3000）    | Ubuntu 18.04.1 |
  | ARM+Atlas 300I 推理卡 （型号3000）    | CentOS 7.6     |

* 软件依赖

  | 软件名称 | 版本  |
  | -------- | ----- |
  | cmake    | 3.5.+ |
  | mxVision | 2.0.4 |
  | Python   | 3.9.2 |
  | OpenCV   | 3.4.0 |
  | gcc      | 7.5.0 |
  | ffmpeg   | 4.3.2 |

## ３ 代码主要目录介绍

本Sample工程名称为Actionrecognition，工程目录如下图所示：

```
.
├── data
│   ├── roi
│   │   ├── Climbup
│   │   └── ...
│   └── video
│   │   ├── Alone
│   │   └── ...
├── models
│   ├── ECONet
│   │   └── ...
│   └── yolov3
│   │   └── ...
├── pipeline
│   ├── plugin_all.pipeline
│   ├── plugin_alone.pipeline
│   ├── plugin_climb.pipeline
│   ├── plugin_outofbed.pipeline
│   ├── plugin_overspeed.pipeline
│   ├── plugin_overstay.pipeline
│   └── plugin_violentaction.pipeline
├── plugins
│   ├── MxpiStackFrame // 堆帧插件
│   │   ├── CMakeLists.txt
│   │   ├── MxpiStackFrame.cpp
│   │   ├── MxpiStackFrame.h
│   │   ├── BlockingMap.cpp
│   │   ├── BlockingMap.h
│   │   └── build.sh
│   ├── PluginAlone // 单人独处插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginAlone.cpp
│   │   ├── PluginAlone.h
│   │   └── build.sh
│   ├── PluginClimb // 攀高检测插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginClimb.cpp
│   │   ├── PluginClimb.h
│   │   └── build.sh
│   ├── PluginOutOfBed // 离床检测插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginOutOfBed.cpp
│   │   ├── PluginOutOfBed.h
│   │   └── build.sh
│   ├── PluginOverSpeed // 快速移动插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginOverSpeed.cpp
│   │   ├── PluginOverSpeed.h
│   │   └── build.sh
│   ├── PluginOverStay // 逗留超时插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginOverStay.cpp
│   │   ├── PluginOverStay.h
│   │   └── build.sh
│   ├── PluginCounter // 计时插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginCounter.cpp
│   │   ├── PluginCounter.h
│   │   └── build.sh
│   ├── PluginViolentAction // 剧烈运动插件
│   │   ├── CMakeLists.txt
│   │   ├── Plugin_ViolentAction.cpp
│   │   ├── Plugin_ViolentAction.h
│   │   └── build.sh
├── main.py
├── README.md
└── run.sh
```

## 4 软件方案介绍

为了完成上述六种应用场景中的行为识别，系统需要检测出同一目标短时间内状态的变化以及是否存在剧烈运动，因此系统中需要包含目标检测、目标跟踪、动作识别与逻辑后处理。其中目标检测模块选取Yolov3，得到行人候选框；目标跟踪模块使用IOU匹配，关联连续帧中的同一目标。将同一目标在连续帧的区域抠图组成视频序列，输入动作识别模块ECONet，模型输出动作类别，判断是否为剧烈运动。逻辑后处理通过判断同一目标在连续帧内的空间位置变化判断难以被定义为运动的其余五种应用场景。系统方案中各模块功能如表1.1 所示。

表1.1 系统方案中个模块功能：

| 序号 | 子系统         | 功能描述                                                     |
| ---- | -------------- | ------------------------------------------------------------ |
| 1    | 初始化配置     | 主要用于初始化资源，如线程数量、共享内存等。                 |
| 2    | 视频流         | 从多路IPC相机拉流，并传输入Device进行计算。                  |
| 3    | 视频解码       | 通过硬件（DVPP）对视频进行解码，转换为YUV数据进行后续处理。  |
| 4    | 图像预处理     | 在进行基于深度神经网络的图像处理之前，需要将图像缩放到固定的尺寸和格式。 |
| 5    | 目标检测       | 基于深度学习的目标检测算法是该系统的核心模块之一，本方案选用基于Yolov3的目标检测。 |
| 6    | 目标跟踪       | 基于卡尔曼滤波与匈牙利算法的目标跟踪算法是该系统的核心模块之一，本方案选用IOU匹配。 |
| 7    | 图像抠图       | 将同一目标在连续帧所在区域抠图，并组成图像序列，输入动作识别模块。 |
| 8    | 动作识别       | 基于深度学习的动作识别算法是该系统的核心模块之一，本方案选用基于ECONet的动作识别模型。 |
| 9    | 单人独处后处理 | 当连续m帧只出现一个目标ID时，则判断为单人独处并报警，如果单人独处报警之前n帧内已经报警过，则不重复报警。 |
| 10   | 快速移动后处理 | 当同一目标在连续m帧中心点平均位移高于阈值v，则判断为快速移动，如果快速移动报警之前n帧内已经报警过，则不重复报警。 |
| 11   | 逗留超时后处理 | 当同一目标在连续m帧中心点平均位移低于阈值v，则判断为快速移动，如果快速移动报警之前n帧内已经报警过，则不重复报警。 |
| 12   | 离床检测后处理 | 当同一目标在连续m帧内从给定区域roi内离开，则判断为离床，如果离床报警之前n帧内已经报警过，则不重复报警。 |
| 13   | 攀高检测后处理 | 当同一目标在连续m帧内从给定区域roi内中心点上升，并且中心点位移大于阈值h，则判断为离床，如果离床报警之前n帧内已经报警过，则不重复报警。 |
| 14   | 剧烈运动后处理 | 动作识别模块输出类别为关注的动作类别时，则判断为剧烈运动，如果剧烈运动之前n帧内已经报警过，则不重复报警。 |

## 5 准备

**步骤1：** 参考安装教程《mxVision 用户指南》安装 mxVision SDK。

**步骤2：** 配置 mxVision SDK 环境变量。

`export MX_SDK_HOME=${安装路径}/mxVision `

注：本例中mxVision SDK安装路径为 /root/work/MindX_SDK/mxVision。

**步骤3：** 推荐在${MX_SDK_HOME}/samples下创建ActionRecognition根目录，在项目根目录下创建目录models `mkdir models`，分别为yolov3和ECONet创建一个文件夹，将两个离线模型及各自的配置文件放入文件夹下。[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ActionRecognition/models.zip)。创建完成后models文件夹下的目录结构如下：

```
.models
├── ECONet
│   ├── eco_aipp.cfg // 模型转换aipp配置文件
│   ├── eco_post.cfg // 模型后处理配置文件
│   ├── ECONet.om // 离线模型
│   ├── ucf101.names // label文件
│   ├── ucf101_best.pb // 冻结pb模型
│   └── trans_pb2om.sh // 模型转换脚本
├── yolov3
│   ├── coco.names // label文件
│   ├── yolov3_tf_bs1_fp16.om // 离线模型
│   └── yolov3_tf_bs1_fp16.cfg // 模型后处理配置文件
```

**步骤4：** 编译程序前提需要先交叉编译好第三方依赖库。

**步骤5：** 配置环境变量MX_SDK_HOME：

```bash
export MX_SDK_HOME=/MindX_SDK/mxVision/								
# 此处MX_SDK_HOME请使用MindX_SDK的实际路径
```

**步骤6**：在插件代码目录下创建build文件夹，使用cmake命令进行编译，生成.so文件。下面以单人独处插件的编译过程作为范例：

```bash
## 进入目录 /plugins/plugin_Alone
## 创建build目录
mkdir build
## 使用cmake命令进行编译
cmake ..
make -j
```

或者使用插件代码目录下的build.sh脚本，例：

```bash
## 前提条件是正确设置export MX_SDK_HOME
chmod +x build.sh
./build.sh
```

编译好的插件会自动存放到SDK的插件库中，可以直接在pipiline中使用。

**步骤7:** 配置pipeline

1.  插件参数介绍

   * MxpiStackFrame

     | 参数名称     | 参数解释             |
     | :----------- | -------------------- |
     | visionSource | 抠图插件名称         |
     | trackSource  | 跟踪插件名称         |
     | frameNum     | 跳帧间隔（为1不跳）  |
     | timeOut      | 某个目标堆帧超时时间 |
     | sleepTime    | 检查线程休眠时间     |
   
   * PluginAlone
   
     | 参数名称            | 参数解释               |
     | :------------------ | ---------------------- |
     | dataSourceDetection | 目标检测后处理插件名称 |
     | dataSourceTrack     | 跟踪插件名称           |
     | detectThresh        | 检测帧数               |
     | detectRatio         | 警报帧阈值             |
     | detectSleep         | 警报间隔               |
   
   * PluginClimb
   
     | 参数名称            | 参数解释               |
     | :------------------ | ---------------------- |
     | dataSourceTrack     | 跟踪插件名称           |
     | dataSourceDetection | 目标检测后处理插件名称 |
     | detectRatio         | 警报帧阈值             |
     | filePath            | ROI配置txt文件         |
     | detectSleep         | 警报间隔               |
     | bufferLength        | 检测帧数窗口大小       |
     | highThresh          | 高度阈值               |
   
   * PluginCounter
   
     | 参数名称           | 参数解释     |
     | :----------------- | ------------ |
     | dataSourceTrack    | 跟踪插件名称 |
     | descriptionMessage | 插件描述信息 |
   
   * PluginOutOfBed
   
     | 参数名称            | 参数解释               |
     | :------------------ | ---------------------- |
     | dataSourceTrack     | 跟踪插件名称           |
     | dataSourceDetection | 目标检测后处理插件名称 |
     | configPath          | ROI配置txt文件         |
     | detectThresh        | 检测帧数窗口大小       |
     | detectSleep         | 警报间隔               |
     | detectRatio         | 警报帧阈值             |
   
   * PluginOverSpeed
   
     | 参数名称            | 参数解释               |
     | :------------------ | ---------------------- |
     | dataSourceTrack     | 跟踪插件名称           |
     | dataSourceDetection | 目标检测后处理插件名称 |
     | speedThresh         | 速度阈值               |
     | frames              | 检测帧数窗口大小       |
     | detectSleep         | 警报间隔               |
   
   * PluginOverStay
   
     | 参数名称            | 参数解释               |
     | :------------------ | ---------------------- |
     | dataSourceTrack     | 跟踪插件名称           |
     | dataSourceDetection | 目标检测后处理插件名称 |
     | stayThresh          | 逗留时间阈值           |
     | frames              | 检测间隔帧数           |
     | distanceThresh      | 逗留范围               |
     | detectRatio         | 警报帧阈值             |
     | detectSleep         | 警报间隔               |
   
   * PluginViolentAction
   
     | 参数名称        | 参数解释              |
     | :-------------- | --------------------- |
     | classSource     | 分类后处理插件名称    |
     | filePath        | 感兴趣动作类别txt文件 |
     | detectSleep     | 警报间隔              |
     | actionThreshold | 动作阈值              |
   
2. 配置范例

   ```
   ## PluginClimb
   "mxpi_pluginclimb0": {
               "props": {
                   "dataSourceTrack": "mxpi_motsimplesort0",
                   "dataSourceDetection": "mxpi_objectpostprocessor0",
                   "detectRatio":  "0.6",
                   "filePath": "./data/roi/Climbup/*.txt",
                   "detectSleep": "30",
                   "bufferLength": "8",
                   "highThresh": "10"
               },
               "factory": "mxpi_pluginclimb",
               "next": "mxpi_dataserialize0"
           }
   ## /*Yolov3*/
   "mxpi_tensorinfer0":{
               "props": {
                   "dataSource": "mxpi_imageresize0",
                   "modelPath": "./models/yolov3/yolov3_tf_bs1_fp16.om"
                   },
               "factory": "mxpi_tensorinfer",
                "next": "mxpi_objectpostprocessor0"
           }，
   "mxpi_objectpostprocessor0": {
               "props": {
                   "dataSource": "mxpi_tensorinfer0",
                   "funcLanguage":"c++",
                   "postProcessConfigPath": "./models/yolov3/yolov3_tf_bs1_fp16.cfg",
                   "labelPath": "./models/yolov3/coco.names",
                   "postProcessLibPath": "../../lib/modelpostprocessors/libyolov3postprocess.so"
               },
               "factory": "mxpi_objectpostprocessor",
               "next": "mxpi_distributor0"
           },
   ## ECONet
   "mxpi_tensorinfer1":{
               "props": {
                   "dataSource": "mxpi_stackframe0",
                   "skipModelCheck": "1",
                   "modelPath": "./models/ECONet/ECONet.om"
                   },
               "factory": "mxpi_tensorinfer",
               "next": "mxpi_classpostprocessor0"
           },
   "mxpi_classpostprocessor0":{
               "props": {
                   "dataSource": "mxpi_tensorinfer1",
                   "postProcessConfigPath": "./models/ECONet/eco_post.cfg",
                   "labelPath":"./models/ECONet/ucf101.names",
                 "postProcessLibPath":"../../lib/modelpostprocessors/libresnet50postprocess.so"
                   },
               "factory": "mxpi_classpostprocessor",
               "next": "mxpi_violentaction0"
          },
   ```

   根据所需场景，配置pipeline文件，调整路径参数以及插件阈值参数。例如"filePath"字段替换为roi/Climb目录下的感兴趣区域txt文件，“postProcessLibPath”字段是SDK模型后处理插件库路径。

3. 将pipeline中“rtspUrl”字段值替换为可用的 rtsp 流源地址（需要自行准备可用的视频流，视频流格式为H264），[自主搭建rtsp视频流教程](###7.3-数据下载与RTSP)。

**步骤8：** 在main.py中，修改pipeline路径、对应的流名称以及需要获取结果的插件名称。

```python
## 插件位置
with open("./pipeline/plugin_outofbed.pipeline", 'rb') as f:
        pipelineStr = f.read()
## pipeline中的流名称
streamName = b'classification+detection'
## 想要获取结果的插件名称
key = b'mxpi_pluginalone0'
```

## 6 运行

修改 run.sh 文件中的环境路径和项目路径。

```bash
export MX_SDK_HOME=${CUR_PATH}/../../..
## 注意当前目录CUR_PATH与MX_SDK_HOME环境目录的相对位置
```

直接运行

```bash
chmod +x run.sh
bash run.sh
```

## 7 常见问题
### 7.1 未配置ROI

#### 问题描述：

攀高检测与离床检测出现如下报错：
```bash
terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(4.2.0) /usr1/workspace/MindX_SDK_Multi_DailyBuild/opensource/opensource-scl7/opencv/modules/imgproc/src/geometry.cpp:103: error: (-215:Assertion failed) total >= 0 && (depth == CV_32S || depth == CV_32F) in function 'pointPolygonTest'
```
#### 解决方案：
在pipeline处没有配置roi区域的txt文件，进行配置即可。

示例：

```json
"mxpi_pluginoutofbed0": {
            "props": {
                "dataSourceTrack": "mxpi_motsimplesort0",
                "dataSourceDetect": "mxpi_objectpostprocessor0",
                "configPath":  "./data/roi/OutofBed/*.txt",
                "detectThresh": "8",
                "detectSleep": "15",
	              "detectRatio": "0.25"
            },
            "factory": "mxpi_pluginoutofbed",
            "next": "mxpi_dataserialize0"
}
```

### 7.2 模型路径配置

#### 问题描述：

检测过程中用到的模型以及模型后处理插件需配置路径属性。

#### 后处理插件配置范例：

```json
"mxpi_objectpostprocessor0": {
	"props": {
    	"dataSource": "mxpi_tensorinfer0",
        "funcLanguage":"c++",
        "postProcessConfigPath": "./models/yolov3/yolov3_tf_bs1_fp16.cfg",
        "labelPath": "./models/yolov3/coco.names",
        "postProcessLibPath": "../../../lib/modelpostprocessors/libyolov3postprocess.so"
    },
    "factory": "mxpi_objectpostprocessor",
    "next": "mxpi_motsimplesort0"
}
```

### 7.3 数据下载与RTSP

H264视频文件及ROI文件：[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ActionRecognition/data.zip) ；

RTSP取流地址（可以从网络摄像机获取，也可通过Live555等工具将本地视频文 件转换为rtsp流）。自主搭建RTSP拉流教程：[live555链接](https://bbs.huaweicloud.com/forum/thread-68720-1-1.html)，需要注意的是在搭建RTSP时，使用./genMakefiles <os-platform>命令生成编译文件时，输入的<os-platform>参数是根据cofig.<后缀>获取的，与服务器架构等有关。

RTSP视频拉流插件配置范例：

```json
"mxpi_rtspsrc0": {
	"props": {
    	"rtspUrl": "rtsp_Url"
    },
    "factory": "mxpi_rtspsrc",
    "next": "mxpi_videodecoder0"
}
```

其中rtsp_Url的格式是 rtsp:://host:port/Data，host:port/路径映射到mediaServer/目录下，Data为视频文件的路径。

RTSP拉流教程：[live555链接](https://bbs.huaweicloud.com/forum/thread-68720-1-1.html)中第七步视频循环推流，按照提示修改cpp文件可以使自主搭建的rtsp循环推流，如果不作更改，则为有限的视频流；同时第六步高分辨率帧花屏，修改mediaServer/DynamicRTSPServer.cpp文件，将OutPacketBuffer::maxSize增大，例如"500000"，避免出现”The input frame data was too large for our buffer“问题，导致丢帧。修改完后，需要重新运行以下命令：

```cmake
./genMakefiles <os-platform>
make
```

### 7.4 运行Shell脚本

在linux平台下运行shell脚本时，例如build.sh/run.sh，出现如下错误：

```bash
build.sh: Line 15: $'\r': command not found
```

是由于不同系统平台之间的行结束符不同，使用如下命令去除shell脚本的特殊字符：

```bash
sed -i 's/\r//g' xxx.sh
```

## 8 模型转换

本项目中用到的模型有：ECONet，yolov3

yolov3模型下载参考华为昇腾社区[ModelZoo](https://www.hiascend.com/zh/software/modelzoo/detail/1/ba2a4c054a094ef595da288ecbc7d7b4)  
使用以下命令进行转换，请注意aipp配置文件名，此处使用的为自带sample中的相关文件（{Mind_SDK安装路径}/mxVision/samples/mxVision/models/yolov3/）
```bash
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"
```

ECONet离线模型转换参考 [昇腾Gitee](https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/econet/ECONet_tf_paper99)：下载冻结pb模型ucf101_best.pb，编辑trans_pb2om.sh文件，将--model 配置为ECONet模型所在目录，--output配置为模型输出路径，--insert_op_conf配置为aipp文件路径，在命令行输入

```bash
chmod +x trans_pb2om.sh
./trans_pb2om.sh
```

完成ECONet模型转换。模型下载或转换完成后，按照目录结构放置模型。
