# 车道线检测

## 1. 介绍

本样例是基于mxBase开发的端到端推理的python应用程序，可在昇腾芯片上对图像中的车道线进行检测，并对检测到的图像中的每一条车道线进行识别，最后将可视化结果保存为图片形式。

本样例的主要处理流程为： appsrc > mxpi_imagedecoder > mxpi_imageresize > mxpi_tensorinfer > mxpi_objectpostprocessor > appsink 

### 1.1 支持产品

本项目以昇腾Atlas310，Atlas310B卡为主要的硬件平台。

### 1.2 支持的版本

本样例配套的CANN版本为[6.3.RC1(310), 6.2.RC1(310B)](https://www.hiascend.com/software/cann/commercial)，MindX SDK版本为[5.0.RC1](https://www.hiascend.com/software/Mindx-sdk)。

MindX SDK安装前准备可参考[《用户指南》](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)。

###  1.3 代码目录结构与说明

本样例工程名称为UltraFastLaneDetection，工程目录如下图所示：

```
├── PostProcess #后处理
  ├── CMakeLists.txt
  ├── LanePostProcess.cpp 
  ├── LanePostProcess.h
  ├── build.sh 
  ├── run.sh
├── imgs # 流程图  
├── model
  ├── aipp_culane.config
  ├── coco.names
  ├── yolov3_tf_bs1_fp16.cfg
├── README.md 
├── build.sh 

├── main_text.py
├── Lane.pipeline

```

### 1.4 技术实现流程图

![技术流程图](https://gitee.com/lemon-wang/mindxsdk-referenceapps/raw/master/contrib/UltraFastLaneDetection/imgs/技术流程图.jpg)

## 2. 环境依赖

环境依赖软件和版本如下表：

| 软件                | 版本                                                         | 说明                                               |
| ------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| mxVision            | [mxVision 5.0.RC1](https://www.hiascend.com/software/Mindx-sdk) | mxVision软件包                                     |
| Ascend-CANN-toolkit | [310使用6.3.RC1，310B使用6.2.RC1](https://www.hiascend.com/software/cann/commercial) | Ascend-cann-toolkit开发套件包  |
| 操作系统            | [Ubuntu 18.04](https://ubuntu.com/)                          | Linux操作系统                                      |
| OpenCV              | 4.3.0                                                        | 用于结果可视化                                     |



在进行模型转换和编译运行前，需设置如下的环境变量：

```shell
. /usr/local/Ascend/ascend-toolkit/set_envv.sh # Ascend-cann-toolkit开发套件包默认安全路径，根据实际安装路径修改
. ${MX_SDK_HOME}/mxVision/set_env.sh # ${MX_SDK_HOME}替换为用户的SDK安装路径. 
```



### 3. 模型转换

模型转换使用的是ATC工具，具体使用教程可参考[《ATC工具使用指南》](https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md)。车道线检测模型转换所需的aipp配置文件均放置在/Ultra-Fast-Lane-Detection/model目录下。
注意若推理芯片为310B，需将atc-env脚本中模型转换atc命令中的soc_version参数设置为Ascend310B1。

### 3.1 车牌检测模型的转换

**步骤1** **模型获取** 将[车道线检测项目原工程](https://github.com/cfzd/Ultra-Fast-Lane-Detection)克隆到**本地**。


**步骤2** **pth转onnx** 使用原工程**export.py**脚本放至**服务器**工程目录下，执行如下命令：

```
python export.py
```

*注：Python = 3.8.3*

*Pytorch = 1.7.0*

*onnx = 1.10.1*

**步骤3** **onnx转om** 将步骤2中转换获得的[onnx模型](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/UltraFastLaneDetection/culane_18.onnx)存放至**服务器端**的Ultra-Fast-Lane-Detection-master/model/目录下，执行如下命令：

```shell
atc --model=./model/path_to_culane_18.onnx --framework=5  --output=./module/out/culane_18_2 --soc_version=Ascend310  --insert_op_conf=./model/aipp_culane.config
```


## 4. 编译与运行

**步骤1** **修改CMakeLists.txt文件** (文件位于PostProcess文件夹)

第**8**行 **set(MX_SDK_HOME ../MindX_SDK/mxVision)** 语句是设置SDK的安装路径，需将其替换为用户实际的SDK安装路径。

第**11**行 **set(LIBRARY_OUTPUT_PATH ../MindX_SDK/mxVision/samples/mxVision/SamplePostProcess)** 语句是设置.so文件的输出路径，需将其替换为自定义输出路径进行替换。


**步骤2** **编译**  执行shell脚本或linux命令对代码进行编译：

```shell
bash build.sh
或
rm -r build # 删除原先的build目录(如果有的话)
mkdir build # 创建一个新的build目录
cd build # 进入build目录
cmake .. # 执行cmake命令，在build下生成MakeFile文件
make -j# 执行make命令对代码进行编译
```

**步骤3** **推理** 请自行准备**jpg/jpeg**格式图像保存在工程目录下并修改图片路径，执行如下命令：

```shell
python3 main_text.py # 自行替换图片名称
```
注意：训练图片尺寸大小默认设置为：**1640×590**图片来源于culane测试集全部为车内视角（https://xingangpan.github.io/projects/CULane.html）
测试图片尺寸无相关要求，建议车道线清晰且为车内行驶视角
















