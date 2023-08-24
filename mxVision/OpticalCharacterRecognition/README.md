# OCR

# 1 简介

## 1.1 背景介绍

本参考设计的目的主要包括以下几个方面：

1.为金融、安防、互联网等客户提供基于Atlas做OCR的参考样例，验证可行性；

2.作为客户基于mxBase开发OCR应用的编程样例（下称Demo），开放源代码给客户参考，降低客户准入门槛；

3.提供新架构下系统优化案例，打造超高性能原型系统，为客户提供选择Atlas的理由。

本Demo模型选择是面向直排文本，不考虑弯曲文本的情况，并选择JPEG作为输入图像格式，实现识别输入图片中的文字功能。本文档提供对OCR Demo实现方案的说明。

## 1.2 支持的产品

本系统采用Atlas 300I Pro, Atlas 300V Pro作为实验验证的硬件平台。

## 1.3 软件方案介绍

软件方案包含OCR的三个环节：文本检测，方向分类，字符识别。其中文本检测和字符识别是必要环节，方向分类为可选环节。

本Demo支持基于Paddle PP-OCR server 2.0的DBNet(检测)和CRNN(识别)模型在310P芯片上进行静态分档推理。为了提高CPU、NPU资源利用率，实现极致性能，Demo采用了流水并行及多路并行处理方案。

### 代码主要目录介绍

本Demo根目录下src为源码目录，现将src的子目录介绍如下：
**注意**：代码目录中的src/AscendBase/Base/Framework/ModuleProcessors/TextDetectionPost/下的clipper.cpp、clipper.hpp为开源第三方模块，Demo中不包含这两个文件，需用户自行下载这两个文件，然后放在对应位置。
clipper.cpp、clipper.hpp文件下载链接：https://udomain.dl.sourceforge.net/project/polyclipping/clipper_ver6.4.2.zip

```
.
├── src
│   └── main.cpp          // OCR主函数
│   └── build.sh
│   └── CMakeLists.txt
│   └── AscendBase
│       ├── Base
│       │   ├── ArgumentParser          // 命令行参数解析模块
│       │   ├── BlockingQueue           // 阻塞队列模块
│       │   ├── ConfigParser            // 配置文件解析模块
│       │   ├── Framework               // 流水并行框架模块
│       │   │   ├── ModuleManagers      // 业务流管理模块
│       │   │   ├── ModuleProcessors    // 功能处理模块
│       │   │   │   ├── CharacterRecognitionPost       // 字符识别后处理模块
│       │   │   │   ├── CommonData       // 各功能模块间共用的数据结构
│       │   │   │   ├── Processors       // 各功能处理模块
│       │   │   │   │   ├── HandOutProcess       // 图片分发模块
│       │   │   │   │   ├── DbnetPreProcess      // Dbnet前处理模块
│       │   │   │   │   ├── DbnetInferProcess    // Dbnet推理模块
│       │   │   │   │   ├── DbnetPostProcess     // Dbnet后处理模块
│       │   │   │   │   ├── ClsPreProcess        // Cls前处理模块
│       │   │   │   │   ├── ClsInferProcess      // Cls推理模块
│       │   │   │   │   ├── ClsPostProcess       // Cls后处理模块
│       │   │   │   │   ├── CrnnPreProcess       // Crnn前处理模块
│       │   │   │   │   ├── CrnnInferProcess     // Crnn推理模块
│       │   │   │   │   ├── CrnnPostProcess      // Crnn后处理模块
│       │   │   │   │   ├── CollectProcess       // 推理结果保存模块
│       │   │   │   ├── Signal       // 程序终止信号处理模块
│       │   │   │   ├── TextDetectionPost       // 文本检测后处理模块
│       │   │   │   │   ├── clipper.cpp
│       │   │   │   │   ├── clipper.hpp
│       │   │   │   │   ├── TextDetectionPost.cpp
│       │   │   │   │   ├── TextDetectionPost.h
│       │   │   │   ├── Utils       // 工具函数模块
│   └── Common
│       ├── EvalScript
│       │   ├── eval_script.py       // 精度测试脚本
│       │   └── requirements.txt     // 精度测试脚本的python三方库依赖文件
│       ├── InsertArgmax
│       │   ├── insert_argmax.py     // ArgMax算子插入模型脚本
│       │   └── requirements.txt     // ArgMax算子插入模型脚本的python三方库依赖文件
│       ├── LabelTrans                     // 数据集标签转换脚本
│   └── data
│       ├── config
│       │   ├── setup.config       // 配置文件
│       ├── models
│       │   ├── cls
│       │   ├── crnn
│       │   ├── dbnet
├── README.md
```

# 2 环境搭建

### 2.1 软件依赖说明

**表2-1** 软件依赖说明

| 依赖软件          | 版本           | 依赖说明
| -------------   |--------------| ---------------------|
| CANN            | 6.3.RC3 | 提供基础acl/himpi接口 |
| mxVision        | 5.0.RC3      | 提供基础mxBase的能力 |


### 2.2 CANN环境变量设置

```bash
. ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
```

### 2.3 mxVision环境变量设置

```bash
. ${MX_SDK_HOME}/set_env.sh
```

# 3 模型转换及数据集获取

## 3.1 Demo所用模型下载地址

Paddle PP-OCR server 2.0模型:

| 名称      | 下载链接 |
| --------------- | -------------- |
| Paddle PP-OCR server 2.0 DBNet   |  https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar|
| Paddle PP-OCR server 2.0 Cls  | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar|
| Paddle PP-OCR server 2.0 CRNN | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar|

识别模型字典文件下载地址：
https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.5/ppocr/utils/ppocr_keys_v1.txt


## 3.2 Demo所用测试数据集下载地址
数据集ICDAR-2019 LSVT下载地址：


| 名称 | 下载链接 |
|---------|----------|
| 图片压缩包1 | https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz|
| 图片压缩包2 | https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz|
| 标注文件 | https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json|


图片压缩包名为 train_full_images_0.tar.gz 与 train_full_images_1.tar.gz

标签文件名为 train_full_labels.json

### 3.2.1 数据集准备

#### 3.2.1.1 数据集下载
命令参考
```
wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_0.tar.gz
wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_images_1.tar.gz
wget https://dataset-bj.cdn.bcebos.com/lsvt/train_full_labels.json
```

#### 3.2.1.2 数据集目录创建
创建数据集目录
```
mkdir -p ./icdar2019/images
```
#### 3.2.1.3 解压图片并移动到对应目录
解压图片压缩包并移动图片到对应目录
```
tar -zvxf ./train_full_images_0.tar.gz
tar -zvxf ./train_full_images_1.tar.gz
mv train_full_images_0/* ./icdar2019/images
mv train_full_images_1/* ./icdar2019/images
rm -r train_full_images_0
rm -r train_full_images_1
```
#### 3.2.1.4 标签格式转换
label文件格式转换为ICDAR2015格式, 转换脚本位于src/Common/LabelTrans/label_trans.py

运行标签格式转换脚本工具需要依赖的三方库如下所示：
**表3-1** label_trans.py依赖python三方库

| 名称 | 版本 |
| --- | --- |
| numpy | =1.22.4 |
| tqdm | =4.64.0 |


格式转换脚本参考如下:
```
python3 ./label_trans.py --label_json_path=/xx/xx/train_full_labels.json --output_path=/xx/xx/icdar2019/
```

## 3.3 pdmodel模型转换为onnx模型
- **步骤 1**   将下载好的paddle模型转换成onnx模型。
  执行以下命令安装转换工具paddle2onnx

  ```
   pip3 install paddle2onnx==0.9.5
  ```
  运行paddle2onnx工具需要依赖的三方库如下所示：

**表3-2** paddle2onnx依赖python三方库

| 名称 | 版本 |
| --- | --- |
| paddlepaddle | 2.3.0 |

PP-OCR server 2.0 DBNet模型转成onnx模型指令参考如下：
```
paddle2onnx --model_dir ./ch_ppocr_server_v2.0_det_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_det_infer.onnx --opset_version 11 --enable_onnx_checker True --input_shape_dict="{'x':[-1,3,-1,-1]}"
```

PP-OCR server 2.0 CRNN模型转成onnx模型指令参考如下：
```
paddle2onnx --model_dir ./ch_ppocr_server_v2.0_rec_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_rec_infer.onnx --opset_version 11 --enable_onnx_checker True --input_shape_dict="{'x':[-1,3,32,-1]}"
```

PP-OCR server 2.0 Cls模型转成onnx模型指令参考如下：
```
paddle2onnx --model_dir ./ch_ppocr_mobile_v2.0_cls_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_mobile_v2.0_cls_infer.onnx --opset_version 11 --enable_onnx_checker True
```

## 3.4 识别模型插入ArgMax算子
- **步骤 2** 使用算子插入工具insert_argmax给字符识别模型（CRNN）插入argmax算子。

转到src/Common/InsertArgmax目录下，执行脚本，此处需要传入两个参数：'model_path',对应ch_ppocr_server_v2.0_rec_infer.onnx模型所在路径；'check_output_onnx'，为是否需要针对输出模型做校验，默认值为True，可选choices=[True, False]

使用算子插入工具insert_argmax插入argmax算子指令参考:
  ```
   python3 insert_argmax.py --model_path /xx/xx/ch_ppocr_server_v2.0_rec_infer.onnx --check_output_onnx True
  ```
**表3-3** 使用自动算子插入工具插入argmax算子。

|参数名称 | 参数含义 | 默认值 | 可选值 |
| --- | --- | --- | --- |
| model_path | 对应ch_ppocr_server_v2.0_rec_infer.onnx模型所在路径 |''|''|
| check_output_onnx | 是否需要针对输出模型做校验 | True|True,False|

转换出来的结果位于'model_path'路径下，命名为'ch_ppocr_server_v2.0_rec_infer_argmax.onnx'的onnx模型文件。

**表3-4** insert_argmax.py脚本依赖python三方库

| 名称 | 版本 |
| --- | --- |
| onnx | >=1.9.0 |
| onnxruntime | >=1.13.1 |

## 3.5 静态分档模型转换

****进行静态分档模型转换前需设置CANN环境变量**** 默认的路径为
```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
```
请以实际安装路径为准。

### 3.5.1 文本检测模型转换
将文本检测onnx模型转换成om模型，转换指令参考src/data/models/dbnet目录下面的对应的atc转换脚本（ONNX模型路径以实际为准）：
```
bash atc_dynamic.sh
```

### 3.5.2 字符识别模型转换
将字符识别onnx模型转换成om模型，转换指令参考src/data/models/crnn目录下面的对应的atc转换脚本（ONNX模型路径以实际为准）：
```
bash atc_dynamic.sh
```
可在该目录下新建static目录，将转换生成的模型可统一放置static目录内，用于配置文件recModelPath字段的配置。
### 3.5.3 分类模型转换
将onnx模型转换成om模型。转换指令参考src/demo/data/models/cls目录下面的对应的atc转换脚本（ONNX模型路径以实际为准）：
```
bash atc.sh
```

# 4 编译运行

## 4.1 系统方案各子模块功能介绍
**表4-1** 系统方案各子模块功能：

| 序号 | 子系统 | 功能描述 |
| --- | --- | --- |
| 1 | 图片分发模块 | 从用户指定的文件夹读取图片，并分发给后续的模块。 |
| 2 | DBNet模型前处理 | 主要负责解码和缩放，在310P芯片上使用dvpp解码。 |
| 3 | DBNet模型推理 | 本模块负责将前处理好的图片输入进检测模型并获得模型推理出的Tensor。 |
| 4 | DBNet模型后处理 | 本模块主要负责将输出的Tensor根据与训练一致的后处理流程将输入图片切割为包含文本的子图。 |
| 5 | Cls模型前处理 | 对dbnet后处理之后切割的子图做resize和归一化操作以及分类模型推理时的batch划分 |
| 6 | Cls模型推理 | 本模块负责将前处理好的图片输入进分类模型并获得模型推理出的Tensor。 |
| 7 | Cls模型后处理 | 将模型输出的Tensor根据与训练一致的后处理流程将需要翻转的子图翻转180度。|
| 8 | Crnn模型前处理 | 对dbnet后处理之后切割的子图做resize和归一化操作以及识别模型推理时的batch划分 |
| 9 | Crnn模型推理 | 本模块负责将前处理好的图片输入进识别模型并获得模型推理出的Tensor。支持动态batch和静态分档的推理。 |
| 10 | Crnn模型后处理 | 将模型输出的Tensor根据字典转换为文字识别的结果。 |
| 11 | 推理结果保存模块 | 保存推理结果，并在推理结束时发送停止信号。 |

## 4.2 配置

运行前需要在 `/src/data/config/setup.config` 配置以下信息

配置程序运行的deviceId，deviceType及模型路径等信息
**注意**：如果输入图片中包含敏感信息，使用完后请按照当地法律法规要求自行处理，防止信息泄露。
配置device
  ```bash
  deviceId = 0 // 进行推理的device的id
  deviceType = 310P //310P
  ```

配置模型路径
  ```bash
  detModelPath = ./data/models/dbnet/dbnet_dy_dynamic_shape.om // DbNet模型路径
  recModelPath = ./data/models/crnn/static // CRNN模型路径，静态分档时仅需输入包含识别模型的文件夹的路径即可
  clsModelPath = ./data/models/cls/cls_310P.om // CLS模型路径
  ```

配置文本识别模型字符标签文件
  ```bash
  dictPath = ./data/models/crnn/ppocr_keys_v1.txt // 识别模型字典文件
  ```

配置识别文字的输出结果路径
  ```bash
  saveInferResult = false // 是否保存推理结果到文件，默认不保存，如果需要，该值设置为true，并配置推理结果保存路径
  resultPath = ./result // 推理结果保存路径
  ```
**注意**：推理结果写文件是追加写的，如果推理结果保存路径中已经存在推理结果文件，推理前请手动删除推理结果文件，如果有需要，提前做好备份。

## 4.3 编译

- **步骤 1** 登录服务器操作后台，安装CANN及mxVision并设置环境变量。

- **步骤 2** 将mxOCR压缩包下载至任意目录，如“/home/HwHiAiUser/mxOCR”，解压。

- **步骤 3** 执行如下命令，构建代码。

  ```
   cd /home/HwHiAiUser/mxOCR/OpticalCharacterRecognition/src/;
   bash build.sh
  ```

  *提示：编译完成后会生成可执行文件“main”，存放在“/home/HwHiAiUser/mxOCR/OpticalCharacterRecognition/src/dist/”目录下。*

  ## 4.4 运行
  **注意 C++ Demo 运行时日志打印调用的是mxVision里面的日志模块，mxVision默认打印日志级别为error，如果需要查看info日志，请将配置文件logging.conf中的console_level值设为0。**
  logging.conf文件路径：mxVison安装目录/mxVision/config/logging.conf

  ### 输入图像约束

  仅支持JPEG格式，图片名格式为前缀+下划线+数字的形式，如xxx_xx.jpg。

  ### 运行程序
  **注意 在模型的档位较多，或者设置并发数过大的情况下，有可能会导致超出device内存。请关注报错信息。**

  执行如下命令，启动程序。

    ```
    ./dist/main -image_path /xx/xx/icdar2019/images/ -thread_num 1 -direction_classification false -config ./data/config/setup.config
    ```

    根据屏幕日志确认是否执行成功。

    识别结果存放在“/home/HwHiAiUser/mxOCR/OpticalCharacterRecognition/src/result”目录下。

*提示：输入./dist/main -h可查看该命令所有信息。运行可使用的参数如表4-2 运行可使用的参数说明所示。*

**表4-2** 运行可使用的参数说明

| 选项 | 意义 | 默认值 |
| --- | --- | --- |
| -image_path | 输入图片所在的文件夹路径 | ./data/imagePath |
| -thread_num | 运行程序的线程数，取值范围1-4，请根据环境内存设置合适值。 | 1 |
| -direction_classification | 是否在检测模型之后使用方向分类模型。 | false |
| -config | 配置文件setup.config的完整路径。 | ./data/config/setup.config |

### 结果展示

OCR识别结果保存在配置文件中指定路径的infer_img_x.txt中（x 为图片id）
每个infer_img_x.txt中保存了每个图片文本框四个顶点的坐标位置以及文本内容，格式如下:
  ```bash
  1183,1826,1711,1837,1710,1887,1181,1876,签发机关/Authority
  2214,1608,2821,1625,2820,1676,2212,1659,有效期至/Dateofexpin
  1189,1590,1799,1606,1797,1656,1187,1641,签发日期/Dateofissue
  2238,1508,2805,1528,2802,1600,2235,1580,湖南/HUNAN
  2217,1377,2751,1388,2750,1437,2216,1426,签发地点/Placeofis
  ```
**注意**：如果输入图片中包含敏感信息，使用完后请按照当地法律法规要求自行处理，防止信息泄露。

# 4.5 动态库依赖说明

Demo动态库依赖可参见代码中“src”目录的“CMakeLists.txt”文件中“target_link_libraries”参数处。

**表4-3** 动态库依赖说明

| 依赖软件 | 说明 | 
| --- | --- |
| libascendcl.so | ACL框架接口，具体介绍可参见ACL接口文档。 |
| libacl_dvpp.so | ACL框架接口，具体介绍可参见ACL接口文档。 |
| libpthread.so | C++的线程库。 |
| libglog.so | C++的日志库。 |
| libopencv_world.so | OpenCV的基本组件，用于图像的基本处理。|
| libmxbase.so | 基础SDK的基本组件，用于模型推理及内存拷贝等。|

# 5 精度测试脚本使用

## 5.1 依赖python三方库安装

精度测试脚本运行依赖的三方库如表5-1所示

**表5-1** 依赖python三方库

| 名称 | 版本 |
| --- | --- |
| shapely | >=1.8.2 |
| numpy | >=1.22.4 |
| joblib | >=1.1.0 |
| tqdm | >4.64.0 |

安装命令
  ```
  pip3 install + 包名
  ```

## 5.2 运行

### 运行程序

- **步骤 1** 将mxOCR压缩包下载至任意目录，如“/home/HwHiAiUser/mxOCR”，解压。
- **步骤 2** 运行ocr Demo生成推理结果文件。
- **步骤 3** 执行如下命令，启动精度测试脚本，命令中各个参数请根据实际情况指定。

  ```
  cd /home/HwHiAiUser/mxOCR/OpticalCharacterRecognition/src/Common/EvalScript;
  python3 eval_script.py --gt_path=/xx/xx/icdar2019/labels --pred_path=/xx/xx/result
  ```

  根据屏幕日志确认是否执行成功。

*运行可使用的参数如表5-2 运行可使用的参数说明所示。*

**表5-2** 运行可使用的参数说明

| 选项 | 意义 | 默认值 |
| --- | --- | --- |
| --gt_path | 测试数据集标注文件路径。 | "" |
| --pred_path | Ocr Demo运行的推理结果存放路径。 | "" |
| --parallel_num | 并行数。 | 32 |