# KeywordDetection

## 1.介绍

本开发样例演示关键词检测 KeywordDetection，供用户参考。
本系统基于昇腾Atlas300卡。
主要分为图片字符检测，图片字符识别，语言模型，关键词判断四个子系统（暂时只支持英文字符）。

### 1.1支持的产品

本系统采用Atlas300-3010作为实验验证的硬件平台，并支持Atlas200RC以及Atlas500的硬件平台.具体产品实物图和硬件参数请参见《Atlas 300 AI加速卡 用户指南（型号 3010）》。由于采用的硬件平台为含有Atlas 300的Atlas 800 AI服务器 （型号3010），而服务器一般需要通过网络访问，因此需要通过笔记本或PC等客户端访问服务器，而且展示界面一般在客户端。

### 1.2支持的版本

支持1.75.T11.0.B116, 1.75.T15.0.B150, 20.1.0
[SDK 2.0.2.B011](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/OCR/Ascend-mindxsdk-mxvision_2.0.2.b011_linux-aarch64.run)

版本号查询方法，在Atlas产品环境下，运行以下命令：

```bash
npu-smi info
```

### 1.3软件方案介绍

软件方案将关键词检测系统划分为图片字符检测，图片字符识别，语言模型，关键词判断四个子系统。子系统功能具体描述请参考 表1.1 系统方案各子系统功能描述。图片字符检测子系统可以实现图像的输入，并对图像中的字符位置进行检测，然后得到字符的坐标信息，并进行仿射变换，图片字符识别子系统可以实现将图片字符检测子系统的结果进行识别，语言模型子系统获取编码字符的embedding，关键词判断子系统通过计算图片字符embedding和关键词embedding的相似度，来判断图片是否有关键词。本方案选择使用ctpn作为图片字符检测模型，使用crnn作为图片字符识别模型，使用bert作为语言模型。系统方案中各模块功能如表1.2 所示。

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统     |                           功能描述                           |
| ---- | ---------- | :----------------------------------------------------------: |
| 1    | 图片字符检测 | 图片字符检测子系统负责在图像中检测到字符位置并进行透视变换，然后送入到图片字符识别子系统中。 |
| 2    | 图片字符识别 | 图片字符识别子系统将得到的检测结果进行识别，并将识别结果送入语言模型子系统中。 |
| 3    | 语言模型 | 图片字符通过语言模型子系统得到图片字符embedding，并将关键词送入语言模型子系统，得到关键词embedding，将图片字符embedding和关键词embedding送入关键词判断子系统中。 |
| 4    | 关键词判断 |若存在字符embedding和关键词embedding的相似度大于设定阈值，则判断图片中存在关键词，并打印单词和其对应的关键词，否则判断不存在关键词。 |

表1.2 系统方案各子模块功能描述：

| 序号 | 子系统         | 功能描述                                                     |
| ---- | -------------- | ------------------------------------------------------------ |
| 1    | 输入图像       | 将图像（JPG格式）通过本地代码输入到pipeline中。              |
| 2    | 输入关键词     | 将关键词文件（txt格式）通过本地代码输入到pipeline中。        |
| 3    | 图像解码       | 通过硬件（DVPP）对图像进行解码，转换为YUV数据进行后续处理。  |
| 4    | 图像放缩       | 由于文本检测模型的输入大小为固定的维度，需要使用图像放缩插件将图像等比例放缩为固定尺寸。 |
| 5    | 检测模型       | 将放缩之后的图像送入到文本检测模型进行检测，并将结果送入到下游后处理插件。本方案选用ctpn进行文本检测。 |
| 6    | 检测模型后处理 | 将检测模型的结果送入到后处理插件中，得到文本检测结果。       |
| 7    | 仿射变换       | 使用仿射变换插件将文本检测结果转换为正常的axis-glign的文本块。 |
| 8    | 图像放缩       | 文字识别模型的输入为固定维度，所以需要将仿射变换的结果进行等比例放缩。 |
| 9    | 文字识别       | 在图像放缩后，将缓存区数据送入文字识别模型。本方案选用crnn进行文本识别。 |
| 10   | 图像字符编码   | 使用语言模型前处理插件对图像字符进行编码。                   |
| 11   | 关键词编码     | 使用语言模型前处理插件对关键词进行编码。                     |
| 12   | 语言模型       | 将图像字符编码结果送入语言模型中获取图像字符embedding。      |
| 13   | 语言模型       | 将关键词编码结果送入语言模型中获取关键词embedding。          |
| 14   | 关键词判断     | 将图像字符embedding和关键词embedding输入关键词判断插件中，进行相似度比较。 |

### 1.4硬件方案简介

考虑到关键词检测系统主要在中心端应用，因此本系统采用Atlas300-3010作为实验验证的硬件平台。Atlas300-3010由独立的4片含有AICore的Hi1910V100芯片组成。由于采用的硬件平台为含有Atlas300-3010的服务器，而服务器一般需要通过网络访问，因此需要通过笔记本或PC等客户端访问服务器，而且展示界面一般在客户端。本系统测试验证所使用的服务器型号为Atlas800-3010型号。 

### 1.5方案架构设计

本方案使用了本地Host模式，本地上传代码之后会对代码和输入进行打包编码并送入到pipeline中，下游插件得到结果之后会从buffer中读取数据并反编码，然后进行插件内部的代码逻辑，处理完毕之后继续使用编码-反编码的模型进行传输，整个方案架构不需要服务器端，全部在本地Client完成。

### 1.6代码主要目录介绍

本Sample工程名称为KeywordDetection，工程目录如下图1.2所示：

```
├── KeywordDetection
│   ├── KeywordDetection.sh
│   ├── KeywordDetection.py
│   ├── bert_key.txt
│   ├── MODEL.md
│   └── README.md
├── pipeline
│   └── KeywordDetection.pipeline
├── models
│   ├── ctpn
│   ├── paddlecrnn
│   └── bert
├── data
│   └── en_text
├── plugins
│   ├── TextInfoPlugin
│   │   ├── CMakeLists.txt
│   │   ├── TextInfoPlugin.cpp
│   │   ├── TextInfoPlugin.h
|   |   └── vocab.txt
│   └── TextSimilarityPlugin
│   │   ├── CMakeLists.txt
│   │   ├── TextSimilarityPlugin.cpp
│   │   └── TextSimilarityPlugin.h

```

## 2.环境依赖

推荐系统为ubuntu 18.04或centos 7.6，环境依赖软件和版本如下表：

| 软件名称 | 版本          |
| -------- | ------------- |
| cmake    | 3.14+         |
| Python   | 3.9.2         |
| protobuf | 3.11.2        |
| g++      | 4.8.5 / 7.3.0 |
| GLIBC    | 2.23          |
| automake | 1.16          |

在编译运行Demo前，需设置环境变量：

*  `ASCEND_HOME`      Ascend安装的路径，一般为 `/usr/local/Ascend`
*  `DRIVER_HOME`      可选，driver安装路径，默认和$ASCEND_HOME一致，不一致时请设置
*  `ASCEND_VERSION`   acllib 的版本号，用于区分不同的版本，参考$ASCEND_HOME下两级目录，一般为 `ascend-toolkit/*version*`
*  `ARCH_PATTERN`     acllib 适用的 CPU 架构，查看```$ASCEND_HOME/$ASCEND_VERSION```文件夹，可取值为 `x86_64-linux` 或 `arm64-linux`等

```bash
export ASCEND_HOME=/usr/local/Ascend
export DRIVER_HOME=/usr/local/Ascend
export ASCEND_VERSION=ascend-toolkit/latest
export ARCH_PATTERN=x86_64-linux
```

## 3.编译

**步骤1**：编译程序前提需要先交叉编译好第三方依赖库，第三方依赖库详见下面目录<软件依赖说明>。

注意：第三方依赖库请使用对应版本的GCC编译，否则会出现编译程序连接失败的问题。

**步骤2**：修改代码目录中：```plugins/```下的各个插件中```CMakeLists.txt```文件中的路径为实际安装依赖包的路径。修改代码路径如下所示：

```cmake
set(MX_SDK_HOME ${XXX}/MindX_SDK/mxVision/)						       # Host侧SDK 请修改为实际路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${XM_SDK_HOME}/lib/plugins) # Host侧插件 请修改为实际路径
```

**步骤3**：编译插件，生成.so文件

```bash
mkdir build
cmake ..
make -j
```

## 4.运行

### 4.1部署

**步骤1**：在官网下载bert词表vocab.txt[bert-base-uncased at main (huggingface.co)](https://huggingface.co/bert-base-uncased/tree/main)，放入目录**./plugins/TextInfoPlugin**下

**步骤2**：[bert模型](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/OCR/model/model_bert.om)，将```bert```模型放到```models/bert```文件夹内，[ctpn模型](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/OCR/model/model_ctpn.zip)，将```ctpn```模型放到```models/ctpn```文件夹内，[crnn模型](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/OCR/model/models_ocr.zip)，将```crnn```模型放到```models/paddlecrnn```文件夹内，具体的模型转换方法见当前目录下的MODEL.md。

**步骤3**：配置环境变量，根据自己的环境变量不同，需要配置不同的环境变量，下面给出参考示例：

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH

export PATH=/usr/local/python3.9.2/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export LD_LIBRARY_PATH=${XXX}/project/opencv/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${XXX}/mxManufacture/lib:${XXX}/mxManufacture/opensource/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH="${XXX}/mxManufacture/python:${PYTHONPATH}"
export MX_SDK_HOME="${XXX}/MindX_SDK/mxVision"
export LD_LIBRARY_PATH="${XXX}/MindX_SDK/mxManufacture/lib:${XXX}/MindX_SDK/mxManufacture/opensource/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}"
export GST_PLUGIN_SCANNER="${XXX}/MindX_SDK/mxManufacture/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${XXX}/MindX_SDK/mxManufacture/opensource/lib/gstreamer-1.0:${XXX}/MindX_SDK/mxManufacture/lib/plugins"
export PYTHONPATH="${XXX}/MindX_SDK/mxManufacture/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${LD_LIBRARY_PATH}"
```

注意：请把${XXX}替换为具体的SDK安装路径

**步骤4**：在```KeywordDetection.py```中，更改```pipeline```路径。

### 4.2 运行

直接运行

```bash
sh KeywordDetection.sh
```

### 4.3 运行结果

运行结果会打印在控制台上，输入图片中是否有关键词，以及所对应关键词。

## 5.软件依赖说明

程序的软件依赖可参见`src`目录的`CMakeLists.txt`文件，见文件末尾“target_link_libraries”参数处。

### 5.1 软件依赖

| 依赖软件 | 版本   | 说明                                 |
| -------- | ------ | ------------------------------------ |
| opencv   | 4.2.0  | OpenCV的基本组件，用于图像的基本处理 |
| protobuf | 3.11.2 | 数据序列化反序列化组件。             |
| ffmpeg   | 4.2.1  | 图像转码解码组件                     |

## 6.常见问题

### 6.1 安装opencv失败

#### 问题描述：

编译失败：

```
undefine reference to 'gst_***'
...
collect2: error: ld returned l exit status
```

#### 解决方案：

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE \
-DENABLE_NEON=OFF \
-DCMAKE_INSTALL_PREFIX=/usr/local/opencv \
-DCMAKE_CXX_FLAGS="-march=armv8-a" \
-DWITH_WEBP=OFF \
-DWITH_GSTREAMER=OFF \
-DBUILD_opencv_world=ON ..
make -j8
```

## 7. 测试数据来源

ICDAR2013数据集

[测试数据](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/OCR/data/en_text.rar)

