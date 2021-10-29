# Pixellink文本检测

## 1 介绍
（项目的概述，包含的功能）
（项目的主要流程）
本开发样例完成图像文本检测功能，供用户参考。本系统基于mxVision SDK进行开发，以昇腾Atlas310卡为主要的硬件平台，开发端到端准确识别图像文本的位置信息，最后能够实现可视化，将识别到的文本位置用线条框选出来。本项目试用场景为：包含文字区域的图像，要求文字区域尽可能清晰，区域大小能够占图像尺寸的20%及以上最佳。图像要求为768x1280x3的彩色图像。文字区域不清晰或者不存在文字区域时，识别可能会出现问题。本项目在运行业务流后，会生成一个txt文件，其中包含文字区域位置的点坐标，每一个文字区域由四个坐标点组成。最后会根据txt文件中的每一组坐标点在图像上绘制出文本区域的线条框。


### 1.1 支持的产品

本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.2 支持的版本

支持的SDK版本，列出版本号查询方式。

支持的SDK版本为2.0.2。

版本号查询方法，在Atlas产品环境下，运行命令：npu-smi info进行查看。


### 1.3 软件方案介绍

本系统设计了不同的功能模块。主要流程为：图片传入流中，将图像放缩至特定尺寸，再利用tensorflow的文本检测模型检测文本区域，生成两个tensor值，将这两个tensor的值进行softmax处理，然后经过后处理完成掩码生成，像素点的并查集构建以及文本框轮廓的获取等。各模块功能描述如表1.1所示：

表1.1 系统方案中各模块功能：
| 序号 | 子系统    | 功能描述                                                                   |
|----|--------|------------------------------------------------------------------------|
| 1  | 图像输入   | 调用MindX SDK的appsrc插件对视频数据进行拉流                                          |
| 2  | 图像放缩   | 调用MindX SDK的mxpi_imageresize                                           |
| 3  | 文本区域检测  | 通过文本区域检测模型，获取图像中的文本位置、以及像素点关联等数据                                           |
| 4  | 结果输出   | 将文本检测结果输出到txt文件中    |
| 5  | 可视化   | 将txt文件中的文本区域坐标信息绘制到图像中    |


### 1.4 代码目录结构与说明

本工程名称为PixelLink，工程目录如下图所示：

```
.
├── models
│   └── convert.cfg  // 模型后处理配置文件
├── pipeline
│   └── Pixel.pipeline
├── get_version.py
├── main.py
├── README.md
├── main_get_groundtruth.py
├── rrc_1_1.py
└── script.py
```


### 1.5 技术实现流程图

![Pixellink文本检测流程图](https://images.gitee.com/uploads/images/2021/1029/112024_3a19c293_9366121.png "屏幕截图.png")



## 2 环境依赖

推荐系统为ubantu 18.04，环境依赖软件和版本如下表：

| 软件名称  | 版本   |
| -------- | ------ |
| cmake    | 3.5+   |
| mxVision | 2.0.2  |
| python   | 3.7.5  |

在编译运行项目前，需要设置环境变量：

模型转换所需ATC工具环境搭建参考链接：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0004.html

在编译运行项目前，需要设置环境变量：

步骤1：安装mxVision SDK。 
步骤2：配置mxVision SDK环境变量、lib库环境变量以及python环境变量。

```
export MX_SDK_HOME=${安装路径}/mxVision
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${MX_SDK_HOME}/python:${PYTHONPATH}"
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins"
```

- 环境变量介绍
- MX_SDK_HOME为SDK安装路径
- LD_LIBRARY_PATH为lib库路径
- PYTHONPATH为python环境路径


## 3 模型转换
本项目中用到的模型有：基于tensorflow转化出的pb形式的pixelLink模型。

pb模型提供在链接链接：https://pan.baidu.com/s/1Avrjhc_J6va3YrGm91GXdQ  提取码：fy4j;

pixellink.om模型下载链接：链接：https://pan.baidu.com/s/1YhrPKZzh_sZQCUfqY9Xxqw  提取码：xhyf;

转换离线模型参考昇腾Gitee：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html。首先需要配置ATC环境，下载pb模型，放到相应的路径后，修改模型转换的cfg配置文件，配置文件已经上传至项目目录models下。使用命令

```
atc --model=pixellink_tf.pb --framework=3 --output=pixellink --output_type=FP32 --soc_version=Ascend310 --input_shape="Placeholder:1,768,1280,3"
--insert_op_conf=convert.cfg --log=info
```
转化项目模型。


注意：转化时，可根据需要修改输出的模型名称。转化成功的模型也同时附在pb模型下载链接中。注意模型以及转化所需文件的路径，防止atc命令找不到相关文件。


## 4 编译与运行
**步骤1**
下载项目文件，以及数据集，其中项目文件链接在模型转换部分已经给出。数据集链接：链接：https://pan.baidu.com/s/107gUYlJP0v4_KlKksJy5pw   提取码：b59b

**步骤2**
在安装mxVision SDK后，配置SDK安装路径、lib路径以及python路径，这些路径需要根据用户实际情况配置，例如SDK安装路径需要与用户本身安装路径一致，不一致将导致环境错误。同理，lib路径与python路径，都需要与实际情况一致。将下载的模型文件以及其他配置文件放到项目路径中，与pipeline内路径对应。修改pipeline内路径与模型文件一致。需要按照代码中的路径去创建文件路径，也可以根据实际需要修改代码中的路径变量。在准备计算指标时，需要人工将代码生成的txt文件压缩到一个zip文件中，并将zip文件和groudtruth的zip文件放到相同路径下，运行评测代码计算指标。


**步骤3** 
将数据集放到项目内，可以从中取出一张图像，命名为test.jpg，并放到与main.py同路径下。

**步骤4**
运行推理代码：

```
python3.7 main.py
```
输出结果：可以直接得到这张测试图像的推理结果，该结果会存到一个txt文件中，并在同目录下可视化test.jpg的检测结果。结果命名为my_test.jpg。

运行评测代码：

将解压后的ch4_test_image数据集放置在与main_get_groundtruth.py同目录下，运行main_get_groundtruth.py，会生成数据集中每张图像的检测结果，检测结果会存放到目标路径下。需要人工将结果压缩为zip文件，命名为om_result.zip，压缩后将zip文件和groundtruth的zip文件放到script.py路径下，gt.zip是原模型的运行结果，用以作为评测的基准。该zip文件可以在链接：https://pan.baidu.com/s/1bo9-ooew4DaOqj-QlAQ_kw  提取码：guea获取。最后，运行script.py，得到评测结果。
```
python3.7 main_get_groundtruth.py
python3.7 script.py --g=./gt.zip --s=./om_result.zip
```
输出结果：首先得到本模型的推理结果，再通过运行脚本代码可以得到原模型输出结果与本模型的结果的对比，最后得到本模型的平均指标。

