# 标题（项目标题）

## 1 介绍

在本系统中，目的是基于MindX SDK，在华为云昇腾平台上，开发端到端工业指针型表计读数的参考设计，实现对传统机械式指针表计的检测与自动读数功能，达到功能要求。

### 1.1 支持的产品

昇腾 310（推理）

### 1.2 支持的版本

本样例配套的 CANN 版本为 5.0.4，MindX SDK 版本为 2.0.4。

MindX SDK 安装前准备可参考《用户指南》，安装教程


### 1.3 软件方案介绍

本系统识别的流程是：先将输入的图像送入流中解码和缩放大小，使用YOLOv5目标检测模型去检测图片中的表盘，结束流。将目标框裁剪下来，再送入流中解码和缩放大小，用DeepLabv3语义分割模型去得到工业表中的指针和刻度，对语义分割模型预测的结果进行读书后处理，找到指针指向的刻度，根据刻度的间隔和刻度根数计算表盘的读数。

表2.1 系统方案中各模块功能：

| 序号 | 子系统 | 功能描述 |
| :------------ | :---------- | :---------- |
| 1    | 图像输入 | 调用MindX SDK的appsrc输入图片|
| 2    | 图像解码 | 调用MindX SDK的mxpi_imagedecoder输入图片|
| 3    | 图像放缩 | 调用MindX SDK的mxpi_imageresize，放缩到1024*576大小 |
| 4    | 工业表检测 | 调用MindX SDK的mxpi_tensorinfer，使用YOLOv5的检测模型，检测出图片中车辆|
| 5    | 保存工业表的图像 | 将YOLOv5检测到的工业表结果保存图片|
| 6    | 图像输入| 调用MindX SDK的appsrc输入检测到的工业表 |
| 7    | 图像解码 | 调用MindX SDK的mxpi_imagedecoder输入图片|
| 8    | 图像放缩 | 调用MindX SDK的mxpi_imageresize，放缩到512*512大小 
| 9    | 指针刻度检测 | 调用MindX SDK的mxpi_tensorinfer，使用DeepLabv3语义分割模型，检测图像中的指针与刻度|
| 10    | 模型后处理 | 调用MindX mxpi_semanticsegpostprocessor，得到语义分割的结果|
| 11    | 读数后处理 | 开发mxpi_process3插件，读出工业表的数字|


### 1.4 代码目录结构与说明

本工程名称为工业指针型表计读数，工程目录如下图所示：

```
│  .bashrc
│  README.md
│  set.sh
│
├─images
│  │  det_res.jpg
│  │  seg_test.jpg
│  │  test.jpg
│  │
│  ├─det_res
│  │      test.jpg
│  │      test0.jpg
│  │      test1.jpg
│  │
│  └─readme
│          det_pipeline.png
│          main_result.png
│          seg_pipeline.png
│
├─infer
│      det_test.py
│      fusion_result.json
│      main.py
│      main.sh
│      seg.py
│
├─models
│  ├─deeplabv3
│  │      seg_aipp.cfg     #deeplabv3的onnx模型转换成om模型的配置文件
│  │
│  └─yolov5
│          det_aipp.cfg     #yolov5的onnx模型转换成om模型的配置文件
│
├─pipeline      #pipeline文件
│  ├─deeplabv3
│  │      deeplabv3.cfg
│  │      deeplabv3.names
│  │      seg.pipeline
│  │
│  └─yolov5
│          det.pipeline
│
├─plugins       #开发读数后处理插件代码
│  └─process3
│          CMakeLists.txt
│          Myplugin.cpp
│          Myplugin.h
│          postprocess.cpp
│          postprocess.h
│          run.sh
│
└─python    #验证精度代码
    ├─deeplabv3_val     #deeplabv3模型测试精度
    │      seg_evaluate.py
    │
    └─yolov5_val        #yolov5模型测试精度
            computer_mAP.py      #4.就算模型的mAP
            mAP_det.py       #1.使用om模型检测测试数据，将得到的结果保存成yolo格式的txt文件
            no_det.py        #3.检测是否有的图像没有目标
            yolo2voc.py      #2.将得到的检测结果yolo数据格式转换成voc格式
```


### 1.5 技术实现流程图
<ol>
<li>基础环境：Ascend 310、mxVision、Ascend-CANN-toolkit、Ascend Driver
<li>模型转换：

PyTorch模型转昇腾离线模型：yolov5.onnx-->yolov5.om

onnx模型转昇腾离线模型：DeepLabv3.onnx  -->  DeepLabv3.om

<li>业务流程编排与配置

<li>yolov5后处理开发

<li>mxpi_process3插件的后处理开发

<li>python推理流程代码开发:


  <center>
      <img src="https://gitee.com/jiangjiang1353/mindxsdk-referenceapps/raw/master/contrib/MeterReader/images/readme/det_pipeline.png">
      <br>
      <div style="color:orange;
      display: inline-block;
      color: #999;
      padding: 2px;">图1. YOLOv5的pipeline流程图 </div>
  </center>
  

  <center>
        <img src="https://gitee.com/jiangjiang1353/mindxsdk-referenceapps/raw/master/contrib/MeterReader/images/readme/seg_pipeline.png">
        <br>
        <div style="color:orange;
        display: inline-block;
        color: #999;
        padding: 2px;">图2. DeepLabv3的pipeline流程图 </div>
  </center>
</ol>


### 1.6 特性及适用场景

在电力能源厂区需要定期监测表计读数，以保证设备正常运行及厂区安全。但厂区分布分散，人工巡检耗时长，无法实时监测表计，且部分工作环境危险导致人工巡检无法触达。针对上述问题，希望通过摄像头拍照后利用计算机智能读数的方式高效地完成此任务。

注：本系统中只使用了两种类型的表盘数据参与训练和测试。我们通过预测的刻度根数来判断表盘类型，第一种表盘的刻度根数为50，第二种表盘的刻度根数为32。因此，目前系统只能实现这两种针表计的检测和自动读数功能。


## 2 环境依赖

### 2.1 环境依赖软件和版本
环境依赖软件和版本如下表：

|   软件名称     |    版本     |
| :-----------: | :---------: |
|    ubuntu     | 18.04.1 LTS |
|   MindX SDK   |    2.0.4    |
|    Python     |    3.9.2    |
|     CANN      |    5.0.4    |
|     numpy     |   1.23.4    |
| opencv-python |    4.6.0    |


### 2.2 基础环境变量——env.sh

```
export MX_SDK_HOME="${SDK安装路径}/mxVision"
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${MX_SDK_HOME}/python:${PYTHONPATH}"
```

### 2.3 ATC工具环境变量——atc_env.sh
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/arm64-linux/atc/ccec_compiler/bin:${install_path}/arm64-linux/atc/bin:$PATH
export PYTHONPATH=${install_path}/arm64-linux/atc/python/site-packages:${install_path}/arm64-linux/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/arm64-linux/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/arm64-linux/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

## 3 模型转换及依赖安装

### 3.1 模型转换

使用模型转换工具 ATC 将 onnx 模型转换为 om 模型，模型转换工具相关介绍参考链接：[CANN 社区版](前言_昇腾CANN社区版(5.0.4.alpha002)(推理)_ATC模型转换_华为云 (huaweicloud.com)) 。

1、YOLOv5模型转换：

下载训练好的onnx模型（下载链接：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MeterReader/models.zip），存放在MeterReader/models/yolov5路径下，文件名为det.onxx,使用命令语句跳转到models/yolov5文件路径下，将模型用以下语句转换成om模型：

  ```bash
  atc --model=det.onnx --framework=5 --output=det  --insert_op_conf=det_aipp.cfg --soc_version=Ascend310 
  ```

执行命令成功后会出现以下语句：
  ```
  ATC start working now, please wait for a moment.
  ATC run success, welcome to the next use.
  ```

出现此语句表示命令执行成功，会在路径下出现det.om模型文件。


2、DeepLabv3模型转换：

在python中安装paddle2onnx库，下载训练好的pdmodel模型(下载链接https://bj.bcebos.com/paddlex/examples2/meter_reader//meter_seg_model.tar.gz)，存放在MeterReader/models/deeplabv3路径下，文件名为meter_seg_model，使用命令语句跳转到models/deeplabv3文件路径下，将模型用以下语句转换成onnx模型：

  ```bash
 paddle2onnx --model_dir saved_inference_model \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file seg.onnx \
            --enable_dev_version True
  ```

或者可以直接下载转换好的onnx模型（下载链接：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MeterReader/models.zip），再使用命令语句跳转到MeterReader/models/deeplabv3文件路径下，使用配置文件和将模型用以下语句转换成om模型：

  ```bash
  atc --model=seg.onnx --framework=5  --output=seg insert_op_conf=seg_aipp.cfg  --input_shape="image:1,3,512,512"  --input_format=NCHW --soc_version=Ascend310
  ```

执行命令成功后会出现以下语句：
  ```
  ATC start working now, please wait for a moment.
  ATC run success, welcome to the next use.
  ```

出现此语句表示命令执行成功，会在路径下出现seg.om模型文件。

### 3.2 依赖安装

说明

* {version}为开发套件包版本号，{arch}为操作系统架构，请用户自行替换。
* 安装路径中不能有空格


安装步骤
<ol>
<li>以软件包的安装用户身份SSH登录安装环境。
<li>将MindX SDK开发套件包上传到安装环境的任意路径下（如：“/home/package”）并用cd命令进入套件包所在路径。
<li>增加对套件包的可执行权限：

```bash
chmod +x Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run
```

<li>执行如下命令，校验套件包的一致性和完整性：

```bash
./Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run --check
```

若显示如下信息，说明套件包满足一致性和完整性：

```
Verifying archive integrity... 100% SHA256 checksums are OK. All good.
```

<li>创建MindX SDK开发套件包的安装路径。

* 若用户想指定安装路径，需要先创建安装路径。以安装路径“/home/work/MindX_SDK”为例：
  ```bash
  mkdir -p /home/work/MindX_SDK
  ```
* 若用户未指定安装路径，软件会默认安装到MindX SDK开发套件包所在的路径。
  
<li>安装MindX SDK开发套件包。

* 若用户指定了安装路径。以安装路径“/home/work/MindX_SDK”为例：
  ```bash
  ./Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run --install --install-path=/home/work/MindX_SDK
  ```
* 若用户未指定安装路径：
  ```bash
  ./Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run --install
  ```
安装完成后，若未出现错误信息，表示软件成功安装于指定或默认路径(/usr/local/Ascend/mindx_sdk/mxVision/)下：
```
Uncompressing ASCEND MINDXSDK RNN PACKAGE 100%
```

<li>环境变量生效。

在当前窗口手动执行以下命令，让MindX SDK的环境变量生效。
```bash
source ~/.bashrc
```
</ol>


## 4 编译与运行

示例步骤如下：

<!-- **步骤1** 文件修改

修改set.sh文件(./set.sh)
```bash
将${SDK安装路径}改为SDK的安装路径
``` -->

**步骤1** 设置环境变量

将${SDK安装路径}更改为SDK安装的路径位置。
```bash
# 执行如下命令，打开.bashrc文件
vi .bashrc

# 在.bashrc文件中添加以下环境变量
export MX_SDK_HOME=${SDK安装路径}

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/

export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner

export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

export PYTHONPATH=${MX_SDK_HOME}/python:$PYTHONPATH

# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

#查看环境变量
env
```

**步骤2** 执行编译

编译插件，在项目目录下执行如下命令
```bash
cd plugins/process3
. run.sh
```


**步骤3** 运行及输出结果

总体运行。输入一张图片，输出得到带有预测表盘计数的图片。
```bash
cd infer
python main.py --ifile ${输入图片路径} --ofile ${输出图片路径}
```
执行结束后，可在命令行内得到yolo模型得到的表盘文件路径，以及 通过后续模型得到的预测表盘度数。并可在设定的${输出图片路径}中查看带有预测表盘计数的图片结果。
<center>
    <img src="https://gitee.com/jiangjiang1353/mindxsdk-referenceapps/raw/master/contrib/MeterReader/images/readme/main_result.png">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">图3. 模型运行输出结果 </div>
</center>

<!-- 2.分步运行

2_1.yolo图像检测模型运行。输入一张图片，输出得到识别到的多个表盘。输入和输出的图片文件名称相同。
```bash
cd infer
python det_test.py --ifile ${输入图片路径} --odir ${输出图片目录}
```

2_2.deeplabv3语义分割模型及插件处理。输入一张表盘的图片（上一步模型所得结果),输出预测的表盘度数。
```bash
cd infer
python seg.py --ifile ${输入图片路径} --ofile ${输出图片路径}
``` -->

**步骤4** 精度测试

分别对yolo模型与deeplabv3模型进行精度测试。

1、YOLOv5模型精度测试。

```bash

#下载目标检测模型验证集voc格式数据到MeterReader/python/yolov5_val/det_val_data路径下，下载链接https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MeterReader/data.zip

# 在命令行中跳转到MeterReader/python/yolov5_val文件路径下
cd MeterReader/python/yolov5_val

#使用下面命令运行脚本，得到当前det.om模型检测验证集数据的yolo格式结果，并保存到以图片命名的txt文件中，保存文件路径为MeterReader/python/yolov5_val/det_val_data/det_sdk_txt。
python mAP_det.py

#使用下面命令运行脚本，将模型检测得到的yolo数据格式转换为voc数据格式，结果保存在MeterReader/python/yolov5_val/det_val_data/det_sdk_txt路径中。
python yolo2voc.py

#使用下面命令运行脚本，检测验证集中的数据是否有无目标文件。
python no_det.py

#使用下面命令运行脚本，计算得到det.om在验证集上的mAP，并保存在MeterReader/python/yolov5_val/det_res路径文件下。
python computer_mAP.py
```
经过测试，YOLOv5模型的mAP为100%。



2、deeplabv3模型精度测试。采取Miou指标评价精度。

```bash
#下载语义分割模型验证集voc格式数据到MeterReader/python/deeplabv3_val/seg_val_img路径下，下载链接https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_seg.tar.gz

cd python/deeplabv3_val/
python seg_evaluate.py
```

输出各个图的Miou指标，并求得平均值作为deeplabv3模型的精度指标。经测试，deeplabv3的模型的Miou为66.53%


## 5 软件依赖说明

无第三方软件依赖。


## 6 常见问题

### 6.1 模型转换问题

当图像进入流后，输出到模型的图像格式为yuv，数据类型unit8，但是om模型的时候yolov5需要输入的图像格式为RGB。

**解决方案：**

在转换模型时必须要在AIPP做色域转换，要不然模型输入不正确。

### 6.2 插件权限问题

**问题描述**

运行pipeline调用第三步的数值处理插件时报错，提示Check Owner permission failed: Current permission is 7, but required no greater than 6.

**解决方案**

将插件的权限调整为默认440（只读）即可
```bash
chmod 440 "$MX_SDK_HOME/lib/plugins/libmxpi_sampleplugin.so"
