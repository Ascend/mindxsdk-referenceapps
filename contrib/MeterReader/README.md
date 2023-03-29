# 工业指针型表计读数项-MeterReader

## 1 介绍

在本系统中，目的是基于MindX SDK，在华为云昇腾平台上，开发端到端工业指针型表计读数的参考设计，实现对传统机械式指针表计的检测与自动读数功能，达到功能要求。

### 1.1 支持的产品

昇腾 310B1（推理）

### 1.2 支持的版本


| 软件名称 | 版本   |
| -------- | ------ |
| python    | 3.9.2     | 
| MindX SDK     |    5.0RC1    |
| CANN | 310使用6.3.RC1<br>310B使用6.2.RC1 |

MindX SDK 安装前准备可参考[《用户指南》安装教程](https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/quick_start/1-1%E5%AE%89%E8%A3%85SDK%E5%BC%80%E5%8F%91%E5%A5%97%E4%BB%B6.md)


### 1.3 软件方案介绍

本系统识别的流程是：先将输入的图像送入流中解码和缩放大小，使用YOLOv5目标检测模型去检测图片中的表盘，结束流。将目标框裁剪下来，再送入流中解码和缩放大小，用DeepLabv3语义分割模型去得到工业表中的指针和刻度，对语义分割模型预测的结果进行读数后处理，找到指针指向的刻度，根据刻度的间隔和刻度根数计算表盘的读数。

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
├── build.sh
├── README.md
├── evaluate    
    ├── deeplabv3_val                #deeplabv3模型测试精度
        ├── seg_evaluate.py                         
    ├── yolov5_val                   #yolov5模型测试精度
        ├── det.py                 #1.使用om模型检测测试数据，将得到的结果保存成yolo格式的txt文件
        ├── match.py               #3.检测是否有的图像没有目标
        ├── yolo2voc.py            #2.将得到的检测结果yolo数据格式转换成voc格式      
├── images
    ├── README_img
        ├── DeepLabv3_pipeline.png
        ├── get_map1.png
        ├── get_map2.png
        ├── get_map3.png
        ├── YOLOv5_pipeline.png        
├── infer
    ├── det.py
    ├── main.py 
    ├── seg.py     
├── models
    ├── deeplabv3
        ├── seg_aipp.cfg            #deeplabv3的onnx模型转换成om模型的配置文件
    ├── yolov5
        ├── det_aipp.cfg           #yolov5的onnx模型转换成om模型的配置文件   
├── pipeline                        #pipeline文件
    ├── deeplabv3
        ├── deeplabv3.cfg
        ├── deeplabv3.names
        ├── seg.pipeline
    ├── yolov5
        ├── det.pipeline   
├── plugins                         #开发读数处理插件代码
    ├── process3
        ├── build.sh
        ├── CMakeLists.txt
        ├── Myplugin.cpp
        ├── Myplugin.h
        ├── postprocess.cpp
        ├── postprocess.h
```

### 1.5 技术实现流程图
<ol>
<li>基础环境：mxVision、Ascend-CANN-toolkit
<li>模型转换：

PyTorch模型转昇腾离线模型：yolov5.onnx  -->  yolov5.om

onnx模型转昇腾离线模型：DeepLabv3.onnx  -->  DeepLabv3.om

<li>业务流程编排与配置

<li>yolov5后处理开发

<li>mxpi_process3插件的后处理开发

<li>python推理流程代码开发:
  YOLOv5的pipeline流程图:
  <center>
      <img src="./images/README_img/YOLOv5_pipeline.png">
      <br>
      <div style="color:orange;
      display: inline-block;
      color: #999;
      padding: 2px;">图1. YOLOv5的pipeline流程图 </div>
  </center>
  DeepLabv3的pipeline流程图:
  <center>
        <img src="./images/README_img/DeepLabv3_pipeline.png">
        <br>
        <div style="color:orange;
        display: inline-block;
        color: #999;
        padding: 2px;">图2. DeepLabv3的pipeline流程图 </div>
  </center>
</ol>


### 1.6 特性及适用场景

在电力能源厂区需要定期监测表计读数，以保证设备正常运行及厂区安全。但厂区分布分散，人工巡检耗时长，无法实时监测表计，且部分工作环境危险导致人工巡检无法触达。针对上述问题，希望通过摄像头拍照后利用计算机智能读数的方式高效地完成此任务。

注意事项：
1. 本系统中只使用了两种类型的表盘数据参与训练和测试。我们通过预测的刻度根数来判断表盘类型，第一种表盘的刻度根数为50，第二种表盘的刻度根数为32。因此，目前系统只能实现这两种针表计的检测和自动读数功能。

2. 本系统要求拍摄图片角度正常，尽可能清晰。如果拍摄图片角度不正常，导致图片模糊，则很难正确读出表数。

3. 本系统采用opencv进行图片处理，要求输入文件均为opencv可处理文件。


## 2 环境依赖

### 2.1 环境依赖软件和版本
环境依赖软件和版本如下表：

|   软件名称     |    版本     |
| :-----------: | :---------: |
|    ubuntu     | 18.04.1 LTS |
|   MindX SDK   |    5.0RC1   |
|    Python     |    3.9.2    |
|     CANN      |    310使用6.3.RC1<br>310B使用6.2.RC1   |
|     numpy     |   1.23.4    |
| opencv-python |    4.6.0    |

MindX SDK开发套件部分可参考[MindX SDK开发套件安装指导](https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/quick_start/1-1%E5%AE%89%E8%A3%85SDK%E5%BC%80%E5%8F%91%E5%A5%97%E4%BB%B6.md)

### 2.2 导入基础环境

```bash
. /usr/local/Ascend/ascend-toolkit/set_env.sh #toolkit默认安装路径，根据实际安装路径修改
. ${SDK_INSTALL_PATH}/mxVision/set_env.sh
```

## 3 模型转换及依赖安装

### 3.1 模型转换

使用模型转换工具 ATC 将 onnx 模型转换为 om 模型，模型转换工具相关介绍参考链接：[CANN 社区版](https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md) 。

下载[onnx模型压缩包](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MeterReader/models.zip)并解压
* 将压缩包中的"det.onnx"模型拷贝至"\${MeterReader代码根目录}/models/yolov5"目录下
* 将压缩包中的"seg.onnx"模型拷贝至"\${MeterReader代码根目录}/models/deeplabv3"目录下：

  注：DeepLabv3模型提供了两种转换方式:

  * 若下载[onnx模型压缩包](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MeterReader/models.zip)，则执行上述步骤后，只需完成DeepLabv3模型转换的步骤二；

  * 若下载[pdmodel模型压缩包](https://bj.bcebos.com/paddlex/examples2/meter_reader//meter_seg_model.tar.gz)，则无需执行上述步骤，参考并完成DeepLabv3模型转换的步骤一和步骤二；

  

**YOLOv5模型转换**

进入"\${MeterReader代码根目录}/models/yolov5"目录，执行以下命令将"det.onnx"模型转换成"det.om"模型:
```bash
atc --model=det.onnx --framework=5 --output=det  --insert_op_conf=det_aipp.cfg --soc_version=Ascend310B1 
```

出现以下语句表示命令执行成功，会在当前目录中得到"det.om"模型文件。
  ```
  ATC start working now, please wait for a moment.
  ATC run success, welcome to the next use.
  ```



**DeepLabv3模型转换**

**（1）步骤一**

使用以下命令安装paddle2onnx依赖，[paddle2onnx安装参考链接](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/docs/zh/compile.md)：
```bash
pip3 install paddle2onnx
```
确保已下载[pdmodel模型压缩包](https://bj.bcebos.com/paddlex/examples2/meter_reader//meter_seg_model.tar.gz)，将目录"meter_seg_model"中的文件解压至"${MeterReader代码根目录}/models/deeplabv3"目录下，进入"deeplabv3"目录，使用以下命令将"pdmodel"模型转换成"onnx"模型,[paddle2onnx模型转换参考链接](https://github.com/PaddlePaddle/Paddle2ONNX/)：
  ```bash
  cd ${MeterReader代码根目录}/models/deeplabv3
  paddle2onnx --model_dir meter_seg_model \
              --model_filename model.pdmodel \
              --params_filename model.pdiparams \
              --save_file seg.onnx \
              --enable_dev_version True
  ```

**（2）步骤二**

进入"\${MeterReader代码根目录}/models/deeplabv3"目录，执行以下命令将"seg.onnx"模型转换成"seg.om"模型
  ```bash
  cd ${MeterReader代码根目录}/models/deeplabv3
  atc --model=seg.onnx --framework=5  --output=seg --insert_op_conf=seg_aipp.cfg  --input_shape="image:1,3,512,512"  --input_format=NCHW --soc_version=Ascend310B1
  ```

出现以下语句表示命令执行成功，会在当前目录中得到seg.om模型文件。
  ```
  ATC start working now, please wait for a moment.
  ATC run success, welcome to the next use.
  ```


## 4 编译与运行

示例步骤如下：

**步骤1** 执行编译

编译插件，在项目目录下执行如下命令
```bash
cd ${MeterReader代码根目录}/plugins/process3
. build.sh
```

**步骤2** 修改pipeline文件中的参数地址
* 修改"${MeterReader代码根目录}/pipeline/yolov5/det.pipeline"第40行处文件的绝对路径，将pipeline中所需要用到的模型路径改为存放模型的绝对路径地址：
  ```python
  40 "modelPath":"${MeterReader代码根目录}/models/yolov5/det.om"
  ```

* 修改"${MeterReader代码根目录}/pipeline/deeplabv3/seg.pipeline"第30、38、39行处文件的绝对路径，将pipeline中所需要用到的模型路径、配置文件地址改为绝对路径地址：
  ```python
  30 "modelPath":"${MeterReader代码根目录}/models/deeplabv3/seg.om"
  38 "postProcessConfigPath":"${MeterReader代码根目录}/pipeline/deeplabv3/deeplabv3.cfg",
  39 "labelPath":"${MeterReader代码根目录}/pipeline/deeplabv3/deeplabv3.names",
  ```

**步骤3** 运行及输出结果

总体运行。输入带有预测表盘的jpg图片，在指定输出目录下输出得到带有预测表盘计数的png图片。
```bash
cd ${MeterReader代码根目录}/infer
python main.py --ifile ${输入图片路径} --odir ${输出图片目录}
```

执行结束后，可在命令行内得到yolo模型得到的表盘文件路径，以及通过后续模型得到的预测表盘度数。并可在设定的${输出图片路径}中查看带有预测表盘计数的图片结果。最后展示的结果图片上用矩形框框出了图片中的表计并且标出了预测的表盘读数。


**步骤4** 精度测试

分别对yolo模型与deeplabv3模型进行精度测试。

1、YOLOv5模型精度测试

步骤一：执行以下命令创建所需要的文件目录
```bash
cd ${MeterReader代码根目录}/evaluate/yolov5_val/
mkdir -p det_val_data/det_val_voc
mkdir -p det_val_data/meter_det
mkdir -p det_val_data/det_val_img
mkdir -p det_val_data/det_sdk_txt
mkdir -p det_val_data/det_sdk_voc
```

步骤二：准备标签文件及推理图片

下载[YOLOv5表计检测数据集](https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_det.tar.gz)并解压到任意目录后，将数据集目录中"test"和"train"目录下的所有图片汇总拷贝至"${MeterReader代码根目录}/evaluate/yolov5_val/det_val_data/meter_det"目录下。

我们提供了样例的[模型验证集标签文件](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MeterReader/data.zip)以供下载测试。

完成下载并解压后，将"data/yolov5/det_val_voc"目录下的文件拷贝至"${MeterReader代码根目录}/evaluate/yolov5_val/det_val_data/det_val_voc"目录下。

然后根据拷贝后目录下样例的txt标签文件名称(txt命名格式为：文件名.txt)在"${MeterReader代码根目录}/evaluate/yolov5_val/det_val_data/meter_det"目录下找到对应名称的jpg图片并拷贝至同级"det_val_img"目录下。

步骤三：预处理数据集

执行以下命令后，将在"${MeterReader代码根目录}/evaluate/yolov5_val/det_val_data/det_sdk_txt"目录下生成当前"det.om"模型检测验证集数据的yolo格式结果，并以图片命名的txt格式保存：
```bash
cd ${MeterReader代码根目录}/evaluate/yolov5_val
python det.py
```

再执行以下命令，将上述得到的yolo数据格式转换成voc数据格式，并保存至"${MeterReader代码根目录}/evaluate/yolov5_val/det_val_data/det_sdk_voc"目录下：
```bash
python yolo2voc.py
```

最后执行以下命令，检测验证集中的数据是否有无目标文件：
```bash
python match.py
```

注意事项：运行脚本之前需要把det_sdk_txt和det_sdk_voc文件夹下的文件清空。

步骤四：精度推理

[登录](https://github.com/Cartucho/mAP/edit/master/main.py)并点击下载[mAP-master.zip](https://codeload.github.com/Cartucho/mAP/zip/refs/heads/master)代码压缩包，上传服务器解压后，将代码包中的"main.py"脚本拷贝至"${MeterReader代码根目录}/evaluate/yolov5_val/"目录下，按照以下步骤修改部分代码:

* 修改main.py第47、48、50行处文件路径
  ```python
  47 GT_PATH = os.path.join(os.getcwd(), 'det_val_data', 'det_val_voc')
  48 DR_PATH = os.path.join(os.getcwd(), 'det_val_data', 'det_sdk_voc')
  49 # # if there are no images then no animation can be shown
  50 IMG_PATH = os.path.join(os.getcwd(), 'det_val_data', 'det_val_img')
  ```

* 修改main.py原第64行代码
  ```python
  60 show_animation = False
  61 if not args.no_animation:
  ```

* 在main.py原第243行添加代码
  ```python
  242 def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
  243 to_show = False
  ```

使用下面命令运行脚本，计算得到det.om在验证集上的mAP。
```bash
python main.py
```

经过测试，YOLOv5模型的mAP为100%。


2、deeplabv3模型精度测试。

执行以下命令，创建所需文件目录：
```
cd ${MeterReader代码根目录}/evaluate/deeplabv3_val/
mkdir seg_val_img
cd seg_val_img
mkdir seg_test_img
mkdir seg_test_img_groundtruth
```

下载[语义分割模型验证集voc格式数据](https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_seg.tar.gz)，解压至"${MeterReader代码根录}/evaluate/deeplabv3_val/seg_val_img"目录下。然后执行以下命令拷贝数据：
```
cp -r ${MeterReader代码根目录}/evaluate/deeplabv3_val/seg_val_img/meter_seg/meter_seg/images/val/. ${MeterReader代码根目录}/evaluate/deeplabv3_val/seg_val_img/seg_test_img/

cp -r ${MeterReader代码根目录}/evaluate/deeplabv3_val/seg_val_img/meter_seg/meter_seg/annotations/val/. ${MeterReader代码根目录}/evaluate/deeplabv3_val/seg_val_img/seg_test_img_groundtruth/
```

采用Miou指标进行精度评价。使用下面命令运行脚本：
```bash
cd ${MeterReader代码根目录}/evaluate/deeplabv3_val/
python seg_evaluate.py
```

输出各个图的Miou指标，并求得平均值作为deeplabv3模型的精度指标。经测试，deeplabv3的模型的Miou为67%。


## 5 软件依赖说明

无第三方软件依赖。


## 6 常见问题

### 6.1 模型转换问题

当图像进入流后，输出到模型的图像格式为yuv，数据类型unit8，但是om模型的时候yolov5需要输入的图像格式为RGB。

**解决方案：**

在转换模型时必须要在AIPP做色域转换，要不然模型输入不正确。

### 6.2 精度推理报错

若运行精度推理时出现如下报错：
```
AttributeError: 'FigureCanvasAgg' object has no attribute 'set_window_title'
```
原因是matplotlib版本变动，修改方式如下：
```
原始代码：fig.canvas.set_window_title(...)

修改后：fig.canvas.manager.set_window_title(...)
```



