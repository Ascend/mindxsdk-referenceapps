# X射线图像焊缝缺陷检测

## 1. 介绍

在本系统中，目的是基于MindX SDK，在昇腾平台上，开发端到端X射线图像焊缝缺陷检测的参考设计，实现对图像中的焊缝缺陷进行缺陷类别识别的功能，并把可视化结果保存到本地，达到功能要求。

样例输入：带有已裁剪出焊缝的jpg图片。

样例输出：框出并标有缺陷类型与置信度的jpg图片。

### 1.1 数据集介绍

GDXray是一个公开X射线数据集，其中包括一个关于X射线焊接图像(Welds)的数据，该数据由德国柏林的BAM联邦材料研究和测试研究所收集。

Welds集中W0003 包括了68张焊接公司的X射线图像。本文基于W0003数据集并在焊接专家的帮助下将焊缝和其内部缺陷标注。

数据集下载地址：https://domingomery.ing.puc.cl/material/gdxray/

注：本系统训练使用的数据是由原png图片转为jpg图片，然后经过焊缝裁剪和滑窗裁剪后输入模型训练，推理时所用的图片是已经经过焊缝裁剪的图片。

### 1.2 支持的产品

本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.3 支持的版本

支持的SDK版本为 2.0.4, CANN 版本为 5.1.1, MindSpore版本为1.8。

版本号查询方法，在Atlas产品环境下，运行命令：

```shell
npu-smi info
```

可以查询支持SDK的版本号。

### 1.4 软件方案介绍

本方案中，会先进行滑窗裁剪处理，然后将处理好的图片通过 appsrc 插件输入到业务流程中，最终根据Faster—RCNN模型识别得到缺陷类别和置信度生成框输出标有缺陷类别与置信度的jpg图片。

系统方案各子系统功能描述：

| 序号 | 子系统         | 功能描述                                                     |
| ---- | -------------- | ------------------------------------------------------------ |
| 1    | 图片输入       | 获取 jpg 格式输入图片                                        |
| 2    | 图片解码       | 解码图片                                                     |
| 3    | 图片缩放       | 将输入图片放缩到模型指定输入的尺寸大小                       |
| 4    | 模型推理       | 对输入张量进行推理                                           |
| 5    | 目标检测后处理 | 从模型推理结果计算检测框的位置和置信度，并保留置信度大于指定阈值的检测框作为检测结果 |
| 6    | 结果输出       | 获取检测结果                                                 |
| 7    | 结果可视化     | 将检测结果标注在输入图片上                                   |

### 1.5 代码目录结构与说明

本工程名称为 Faster_R-CNN，工程目录如下所示：

```
.
├── images                               # README文件内的图片
│   ├── 1.png
│   ├── 2.png
│   ├── permissionerror.png
│   ├── W0003_0001.jpg                   # 测试图片
├── postprocess                          # 后处理插件
│   ├── build.sh                         # 后处理编译所需环境变量可参考该文件
│   ├── CMakeLists.txt
│   ├── FasterRcnnMindsporePost.cpp
│   └── FasterRcnnMindsporePost.h
├── python
│   ├── Main
│   │   ├── main.py
│   │   ├── eval.py
│   │   ├── infer.py
│   │   ├── eval_by_sdk.py
│   │   ├── config.py
│   │   ├── util.py
│   │   ├── draw_predict.py
│   │   └── postprocess.py
│   ├── models
│   │   ├── aipp-configs               （需创建）
│   │   │   ├── aipp.cfg               # sdk做图像预处理aipp配置文件
│   │   │   └── aipp_rgb.cfg           # opencv做图像预处理aipp配置文件
│   │   ├── conversion-scripts         # 转换前后模型所放的位置   
│   │   ├── convert_om.sh              # 模型转换相关环境变量配置可参考该文件
│   │   ├── coco2017.names             # 支持的缺陷类别
│   │   ├── fasterrcnn_coco2017.cfg    # 高性能要求配置
│   │   └── fasterrcnn_coco2017_acc_test.cfg         # 高精度要求配置             
│   ├── data                           （需创建）
│   │   ├── test                       # 用于测试该系统功能的数据集目录
│   │   │   ├── infer_result           # 小图(滑窗裁剪后的图片)推理结果所在目录
│   │   │   ├── draw_result            # 最终推理结果的可视化
│   │   │   ├── cut                    # 测试图片所在目录     （需创建）
│   │   │   ├── crop                   # 滑窗裁剪后的小图片所在目录
│   │   │   ├── img_txt                # 小图推理结果txt格式
│   │   │   ├── img_huizong_txt        # 还原到焊缝图片上的未经过nmx去重的标注框信息
│   │   │   └── img_huizong_txt_nms    # 最终推理结果标注框信息(txt格式)
│   │   ├── eval                       # 用于精度测试的数据集  （需创建）
│   │   │   ├── cocodataset            # 进行验证时数据的coco格式
│   │   │   ├── VOCdevkit              # 进行验证时数据的VOC格式
│   └── pipeline
│       ├── fasterrcnn_ms_dvpp.pipeline           # sdk做图像预处理
│       └── fasterrcnn_ms_acc_test.pipeline       # opencv做图像预处理
└── README.md
```

注：验证时有COCO和VOC两种数据格式是因为原图片经过滑窗裁剪后的小图片是以coco的数据格式进行训练的，而本系统最终采用的验证方式是，将经过推理后得到的小图片的标注框信息还原到未经过滑窗裁剪的图片上，再进行VOC评估。

### 1.6 技术实现流程图

<center>
    <img src="./images/1.png">
    <br>
</center>
### 1.7 特性及适用场景

经过测试，在现有数据集的基础上，该项目检测算法可以检测八种焊缝缺陷：气孔、裂纹、夹渣、未熔合、未焊透、咬边、内凹、成形不良，关于缺陷召回率和MAP分数在后续内容中将会提到。本项目属于工业缺陷中焊缝缺陷检测领域，主要针对DR成像设备（数字化X射线成像设备）拍摄金属焊接处成像形成的焊接X射线图像进行缺陷检测。

## 2. 环境依赖

推荐系统为ubuntu 18.04，环境依赖软件和版本如下表：

|   软件名称    |   版本   |
| :-----------: | :------: |
|    ubantu     |  18.04   |
|   MindX SDK   |  2.0.4   |
|    Python     |  3.9.2   |
|     CANN      | 5.1.RC1  |
|     numpy     |  1.23.3  |
| opencv-python | 4.6.0.66 |
|  pycocotools  |  2.0.5   |
|     mmcv      |  1.7.0   |

确保环境中正确安装mxVision SDK。

在编译运行项目前，需要设置环境变量：

MindSDK 环境变量:

```shell
. ${SDK-path}/set_env.sh
```

CANN 环境变量：

```shell
. ${ascend-toolkit-path}/set_env.sh
```

- 环境变量介绍

```
SDK-path: mxVision SDK 安装路径。
ascend-toolkit-path: CANN 安装路径。
```

## 3. 模型转换

本项目中采用的模型是 Faster—RCNN模型，参考实现代码：https://www.hiascend.com/zh/software/modelzoo/models/detail/C/8d8b656fe2404616a1f0f491410a224c/1


1. 将训练好的模型  [fasterrcnn_mindspore.air](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/Faster-RCNN/fasterrcnn_mindspore.air)  下载至 ``python/models/conversion-scripts`` 文件夹下。


2. 将该模型转换为om模型，具体操作为： ``python/models`` 文件夹下,执行指令进行模型转换：

### DVPP模型转换

```
bash convert_om.sh conversion-scripts/fasterrcnn_mindspore.air aipp-configs/aipp.cfg conversion-scripts/fasterrcnn_mindspore_dvpp
```

### OPENCV模型转换

```
bash convert_om.sh conversion-scripts/fasterrcnn_mindspore.air aipp-configs/aipp_rgb.cfg conversion-scripts/fasterrcnn_mindspore_rgb
```

**注**：转换后的OPENCV模型会用OpenCV对图片做预处理，然后进行推理，用户可自行进行选择。

## 4. 编译与运行

**步骤1** 编译后处理插件

切换到``postprocess``目录下，执行命令：

```shell
bash build.sh
```

注：运行完成后需修改当前目录下`./build/libfasterrcnn_mindspore_post.so`文件权限为640。

**步骤2** 准备测试图片

在``python/data/test/cut/``目录下放好待检测的焊缝图片（``./images``下有一张测试图片W0003_0001.jpg）

**步骤3** 图片检测

切换到``python/Main``目录下，执行命令：

```python
python3 main.py
```

命令执行成功后在目录``python/data/test/draw_result``下生成检测结果文件 。

**步骤4** 精度测试

1. 准备精度测试所需图片，将[验证集](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps/contrib/Faster-RCNN/eval.zip)下载到`python/data/eval/`目录下并解压。

1. 打开`python/pipeline/fasterrcnn_ms_dvpp.pipeline`文件，将第45行（postProcessConfigPath）配置参数改为`../models/fasterrcnn_coco2017_acc_test.cfg`。

1. 使用dvpp模式对图片进行推理，切换到``python/Main``目录下，执行命令：

   ```python
   python3 main.py --img_path ../data/eval/cocodataset/val2017/ --pipeline_path ../pipeline/fasterrcnn_ms_dvpp.pipeline --model_type dvpp --infer_mode eval --ann_file ../data/eval/cocodataset/annotations/instances_val2017.json
   ```

2. 因为涉及到去重处理，每种缺陷需要分开评估精度，切换到``python/Main``目录下，执行命令：

   ```python
   # 验证气孔精度
   python3 eval.py --cat_id 1 --object_name "qikong"
   
   # 验证裂纹精度
   python3 eval.py --cat_id 2 --object_name "liewen"
   ```
   
   **注**：cat_id为缺陷标签，object_name为对应缺陷名称，在 ``python/models/coco2017.names``可查看缺陷类别。
   
   | 缺陷种类 |   AP   |
   | :------: | :----: |
   |   气孔   | 0.7251 |
   |   裂纹   | 0.7597 |

## 5. 常见问题

### 5.1 后处理插件权限问题

运行检测 demo 和评测时都需要将生成的Faster_R-CNN后处理动态链接库的权限修改，否则将会报权限错误，如下图所示：

<center>
    <img src="./images/permissionerror.png">
    <br>
</center>
**解决方案**：

切换到``postprocess``目录下，修改`./build/libfasterrcnn_mindspore_post.so`文件权限为640。

### 5.2 模型转换问题

运行模型转换命名后报错：

<center>
    <img src="./images/3.png">
    <br>
</center>
环境变量设置时加了“$”符号。

**解决方案：**

参考build.sh中环境变量设置，并去掉“$”符号。
