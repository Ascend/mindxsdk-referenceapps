# 通用文字识别（中英文）

## 1 简介

通用文字识别样例基于mxVision SDK进行开发，以昇腾Atlas300卡为主要的硬件平台，主要支持以下功能：

1. 图片读取解码：本样例支持JPG及PNG格式图片，采用OpenCV进行解码、缩放等预处理。
2. 文本检测：在输入图片中检测出文本框，本样例选用基于DBNet的文本检测模型，能达到快速精准检测。
3. 投射变换：将识别的四边形文本框，进行投射变换得到矩形的文本小图。
4. 竖排文字旋转：根据文本框的高宽比，大于阈值（默认为1.5），将文本框旋转90°，从竖排文本转换为横排文本。
5. 文本方向检测：识别文本小图上文本的方向--[0°，180°]，如果为180°，则将文本小图进行180°旋转，本样例选用Mobilenet为方向识别模型。
6. 文字识别：识别文本小图上中英文，本样例采用CRNN模型进行文字识别，能够识别中英文.


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
|-- mxVision
|   |-- AllObjectsStructuring
|   |-- GeneralTextRecognition
|   |   |-- C++
|   |   |   |-- CMakeLists.txt
|   |   |   |-- mainMultiThread.cpp
|   |   |   `-- run.sh
|   |   |-- License.md
|   |   |-- README.md
|   |   |-- THIRD PARTY OPEN SOURCE SOFTWARE NOTICE.md
|   |   |-- data
|   |   |   |-- OCR.pipeline
|   |   |   |-- OCR_multi3.pipeline
|   |   |   |-- config
|   |   |   |   |-- cls
|   |   |   |   |   |-- cls.cfg
|   |   |   |   |   `-- ic15.names
|   |   |   |   |-- det
|   |   |   |   |   `-- det.cfg
|   |   |   |   `-- rec
|   |   |   |       `-- rec_cfg.txt
|   |   |   `-- model
|   |   |       |-- MODEL.md
|   |   |       |-- cls_aipp.cfg
|   |   |       |-- det_aipp.cfg
|   |   |       `-- rec_aipp.cfg
|   |   |-- main_ocr.py
|   |   `-- src
|   |       |-- Clipper
|   |       |   `-- CMakeLists.txt
|   |       |-- DBPostProcess
|   |       |   |-- CMakeLists.txt
|   |       |   |-- DBPostProcess.cpp
|   |       |   `-- DBPostProcess.h
|   |       `-- README.md
```

## 4 准备

**步骤1：** 参考安装教程《mxVision 用户指南》安装 mxVision SDK。

**步骤2：** 配置 mxVision SDK 环境变量。

`export MX_SDK_HOME=${安装路径}/mxVision `

注：本例中mxVision SDK安装路径为 /root/MindX_SDK。

**步骤3：** 编译DBNet模型的后处理动态链接库，请根据./src/README.md, 编译相应的动态链接库。

**步骤4：** 准备模型，根据./data/model/MODEL.md文件转换样例需要的模型,并将模型保存到./data/model/目录下。

**步骤5：** 下载文字识别模型的[字典](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/ppocr/utils/ppocr_keys_v1.txt), 由于样例使用的CRNN模型，对应的字典从1开始，0代表为空，请在下载的字典首行添加一行"blank"，并将修改后的字典保存到./data/config/rec/ppocr_keys_v1.txt, ：

```bash
blank
'
疗
绚
诚
娇
.
.
.
```

**步骤6：** 修改配置根目录下的配置文件./data/OCR.pipeline文件：

1. 将所有“deviceId”字段值替换为实际使用的device的id值，可用的 device id 值可以使用如下命令查看：

    `NPU-smi info`

2. 文本检测使用的DBNet，后处理由步骤三编译得到，默认生成到"./lib/libDBpostprocess.so", 如有修改，请修改./data/OCR.pipeline的对应配置：
   
    ```bash
        "mxpi_textobjectpostprocessor0": {
          "props": {
            "postProcessConfigPath": "./data/config/det/det.cfg",
            "postProcessLibPath": "./lib/libDBpostprocess.so"
           },
          "factory": "mxpi_textobjectpostprocessor",
          "next": "mxpi_warpperspective0"
        },
    ```

3. 文本方向检测和文字识别的后处理在mxVision SDK安装目录，本例中mxVision SDK安装路径为 /root/MindX_SDK，如有修改，请修改./data/OCR.pipeline的对应配置：
   
    ```bash
        "mxpi_classpostprocessor0": {
          "props": {
            "dataSource": "mxpi_tensorinfer1",
            "postProcessConfigPath": "./data/configs/cls/cls.cfg",
            "labelPath": "./data/configs/cls/ic15.names",
            "postProcessLibPath": "/root/MindX_SDK/lib/modelpostprocessors/libresnet50postprocess.so"
          },
          "factory": "mxpi_classpostprocessor",
          "next": "mxpi_rotation1:1"
        },
        .
        .
        .
        "mxpi_textgenerationpostprocessor0": {
          "props": {
            "dataSource": "crnn_recognition",
            "postProcessConfigPath": "./data/config/rec/rec_cfg.txt",
            "labelPath": "./data/config/rec/ppocr_keys_v1.txt",
            "postProcessLibPath": "/root/MindX_SDK/lib/modelpostprocessors/libcrnnpostprocess.so"
          },
          "factory": "mxpi_textgenerationpostprocessor",
          "next": "mxpi_dataserialize0"
        },
    ```

**步骤7：** 准备测试图片，在根目录下创建input_data目录，并将包含中英文的JPG或PNG图片拷贝到input_data目录下：

## 5 运行

- 简易python运行样例
  - `python3 main_ocr.py`
<br>
- 多线程高性能c++样例：输入与输出解耦，多线程发送与读取数据；
  - 源文件 `./C++/mainMultiThread.cpp`, 配置 `./data/OCR_multi3.pipeline` 和 `input_data`
  - 运行 `bash run.sh`
