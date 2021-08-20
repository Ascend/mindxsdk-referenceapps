# Individual attribute recognition

## 1 介绍
本开发样例完成个体属性识别功能，供用户参考。本系统基于mxVision SDK进行开发，以昇腾Atlas310卡为主要的硬件平台，开发端到端准确识别多种人脸属性信息，包括年龄、性别、颜值、情绪、脸型、胡须、发色、是否闭眼、是否配戴眼镜、人脸质量信息及类型等。

### 1.1 支持的产品

本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.2 支持的版本

支持的SDK版本为2.0.2。

版本号查询方法，在Atlas产品环境下，运行命令：npu-smi info进行查看。


### 1.3 软件方案介绍

本系统设计了不同的功能模块。主要流程为：图片传入流中，利用Yolov4的检测模型检测人脸，将检测出人脸的图像放缩至特定尺寸，再利用提取关键点的模型进行关键点提取，获取人脸关键点后进行对齐，将对齐结果输入到人脸属性识别模型中，最后以序列化方式输出识别结果。各模块功能描述如表1.1所示：

表1.1 系统方案中各模块功能：
| 序号 | 子系统    | 功能描述                                                                   |
|----|--------|------------------------------------------------------------------------|
| 1  | 图像输入   | 调用MindX SDK的appsrc插件对视频数据进行拉流                                          |
| 2  | 人脸检测   | 利用yolov4的检测模型，检测出图片中人脸                                                 |
| 3  | 图像放缩   | 调用MindX SDK的mxpi_imageresize                                           |
| 4  | 关键点提取  | 通过人脸关键点提取模型，获取人脸图片中的人脸关键点数据。                                           |
| 5  | 人脸对齐   | 通过人脸关键点实现人脸对齐                                                          |
| 6  | 人脸属性识别 | 通过人脸属性识别模型对人脸对齐后的图片提取人脸属性，选取的模型为caffe框架下的FaceAttribute-FAN，需要使用转换工具转化。 |
| 7  | 结果输出   | 将人脸属性识别的结果序列化输出。                                                       |

### 1.4 代码目录结构与说明

本工程名称为Individual，工程目录如下图所示：

```
.
├── data
│   ├── roi
│   │   ├── Climbup
│   │   └── ...
│   └── video
│   │   ├── Alone
│   │   └── ...
├── models
│   ├── ECONet
│   │   └── ...
│   └── yolov3
│   │   └── ...
├── pipeline
│   ├── plugin_all.pipeline
│   ├── plugin_alone.pipeline
│   ├── plugin_climb.pipeline
│   ├── plugin_outofbed.pipeline
│   ├── plugin_overspeed.pipeline
│   ├── plugin_overstay.pipeline
│   └── plugin_violentaction.pipeline
├── plugins
│   ├── MxpiStackFrame // 堆帧插件
│   │   ├── CMakeLists.txt
│   │   ├── MxpiStackFrame.cpp
│   │   ├── MxpiStackFrame.h
│   │   ├── BlockingMap.cpp
│   │   ├── BlockingMap.h
│   │   └── build.sh
│   ├── PluginAlone // 单人独处插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginAlone.cpp
│   │   ├── PluginAlone.h
│   │   └── build.sh
│   ├── PluginClimb // 攀高检测插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginClimb.cpp
│   │   ├── PluginClimb.h
│   │   └── build.sh
│   ├── PluginOutOfBed // 离床检测插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginOutOfBed.cpp
│   │   ├── PluginOutOfBed.h
│   │   └── build.sh
│   ├── PluginOverSpeed // 快速移动插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginOverSpeed.cpp
│   │   ├── PluginOverSpeed.h
│   │   └── build.sh
│   ├── PluginOverStay // 逗留超时插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginOverStay.cpp
│   │   ├── PluginOverStay.h
│   │   └── build.sh
│   ├── PluginCounter // 计时插件
│   │   ├── CMakeLists.txt
│   │   ├── PluginCounter.cpp
│   │   ├── PluginCounter.h
│   │   └── build.sh
│   ├── PluginViolentAction // 剧烈运动插件
│   │   ├── CMakeLists.txt
│   │   ├── Plugin_ViolentAction.cpp
│   │   ├── Plugin_ViolentAction.h
│   │   └── build.sh
├── main.py
├── README.md
└── run.sh
```



### 1.5 技术实现流程图

![个体属性识别流程图](https://images.gitee.com/uploads/images/2021/0819/151524_0f54a517_9366121.png "屏幕截图.png")





## 2 环境依赖

推荐系统为ubantu 18.04，环境依赖软件和版本如下表：
| 软件名称     | 版本    |
|----------|-------|
| cmake    | 3.5.+ |
| mxVision | 2.0.2 |
| python   | 3.7.5 |
| ATC      |       |

atc相关的工具 待补充.........


在编译运行项目前，需要设置环境变量：

步骤1：安装mxVision SDK。
步骤2：配置mxVision SDK环境变量、lib库环境变量以及python环境变量。

```
exprot MX_SDK_HOME=${安装路径}/mxVision
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${MX_SDK_HOME}/python:${PYTHONPATH}"
```

- 环境变量介绍
- MX_SDK_HOME为SDK安装路径
- LD_LIBRARY_PATH为lib库路径
- PYTHONPATH为python环境路径


## 编译与运行
（描述项目安装运行的全部步骤，，如果不涉及个人路径，请直接列出具体执行命令）

**步骤1** （修改相应文件）
下载项目文件，以及数据集，其中项目文件里的部分文件获取链接：https://pan.baidu.com/s/1LolBqYrszngc3y3xhAeXTQ 提取码：sxho。数据集链接：https://pan.baidu.com/s/1_HhMLN73PX78fSrLPGqK1w  提取码:u4cy。

**步骤2** （设置环境变量）
在安装mxVision SDK后，配置SDK安装路径、lib路径以及python路径，将下载的模型文件放到项目路径中，与pipeline内路径对应。

**步骤3** （执行编译的步骤）
将数据集放到项目内，可以从中取出一张图像，命名为test.jpg，并放到与main.py同路径下。

**步骤4** （运行及输出结果）
运行推理代码：

```
python3.7 main.py
```
输出结果：可以直接得到这张测试图像的推理结果

运行评测代码：

```
python3.7 attr_main.py
python3.7 cal_accuracy.py --gt-file=./test_full.txt --pred-file=./img_result.txt
```
输出结果：首先得到本模型的推理结果，再通过运行脚本代码可以得到原模型输出结果与本模型的结果的对比，最后得到本模型的平均指标。


## 5 常见问题

### 5.1 模型路径配置

**问题描述：检测过程中用到的模型以及模型后处理插件需要配置路径属性。

后处理插件配置范例：
```
 "mxpi_classpostprocessor1": {
               "props": {
                "dataSource": "face_attribute",
                "postProcessConfigPath": "./models/resnet50_aipp_tf.cfg",
                "labelPath": "./models/attr.names",
                "postProcessLibPath": "./models/libsamplepostprocess.so"
            },
            "factory": "mxpi_classpostprocessor",
            "next": "mxpi_dataserialize0"
        }
```


## 6 模型转换
本项目中用到的模型有：yolov4，face_quality_0605_b1.om，resnet50

yolov4模型转换及下载参考华为昇腾社区https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/ImageDetectionSample/python。转化完的模型已经提供在链接https://pan.baidu.com/s/1LolBqYrszngc3y3xhAeXTQ 提取码：sxho；

face_quality_0605_b1.om模型已提供在项目目录models下；

resnet50模型下载链接：https://pan.baidu.com/s/1LolBqYrszngc3y3xhAeXTQ 提取码：sxho。转换离线模型参考昇腾Gitee：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html。首先需要配置ATC环境，下载caffemodel以及prototxt文件等，放到相应的路径后，修改模型转换的cfg配置文件，cfg配置文件已经上传至项目目录models下。使用命令

```
atc --input_shape="data:1,3,224,224" --weight="single.caffemodel" --input_format=NCHW --output="simple" --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --framework=0 --model="single.prototxt" --output_type=FP32
```
转化项目模型。

使用命令
```
atc --input_shape="data:1,3,224,224" --weight="single.caffemodel" --input_format=NCHW --output="simple" --soc_version=Ascend310 --insert_op_conf=./insert_op1.cfg --framework=0 --model="single.prototxt" --output_type=FP32
```
转化评测所需模型。

注意：转化时，可根据需要修改输出的模型名称。转化成功的模型也在resnet50模型下载链接中。

