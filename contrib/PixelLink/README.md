# Pixellink文本检测

## 1 介绍
  本开发样例完成图像文本检测功能，供用户参考。本系统基于mxVision SDK进行开发，以昇腾Atlas310卡为主要的硬件平台，开发端到端准确识别图像文本的位置信息，最后能够实现可视化，将识别到的文本位置用线条框选出来。
  本项目试用场景为：包含字母/数字文本区域的场景图像，对于非数据集的场景下，识别要求文字区域尽可能清晰，区域大小能够占图像尺寸的5%及以上最佳，最大不得超过图片的50%左右。图像中文本区域要不存在遮挡、具有较为稀疏的密集度、文本要求为数字或者英文，文本区域的文本字体最好是规范的单个字体。由于原模型的训练场景原因，本样例无法保证对其他类型文本识别的精度，例如屏幕截图等。若文本区域存在遮挡、文本区域过于密集等可能存在无法检测出结果的问题。对于输入图像，图像要求为3通道的的RGB图像。输入其他格式的图像将无法完成检测功能。图像大小为720x1280x3最佳。图像中全部的文字区域不清晰、大小不支持识别、或者不存在英文或数字的文字区域时，识别可能会出现无文本区域的问题，最后可视化即为原图。本项目在运行代码后，会根据测试图像生成一个txt文件，其中包含文字区域位置的点坐标，每一个文字区域由四个坐标点组成。最后会根据txt文件中的每一组坐标点在图像上绘制出文本区域的四边形框。其中，四边形线条框不一定为规则矩形。


### 1.1 支持的产品

本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.2 支持的版本

支持的SDK版本，列出版本号查询方式。

支持的SDK版本为2.0.2。

版本号查询方法，在Atlas产品环境下，运行命令：npu-smi info进行查看。


### 1.3 软件方案介绍

  本系统设计了不同的功能模块。主要流程为：图片传入流中，将图像放缩至特定尺寸，再利用基于tensorflow的pixellink文本检测模型检测文本区域，生成两个tensor值，将这两个tensor的值分别进行softmax处理，然后经过后处理完成掩码生成，像素点的并查集构建以及文本框轮廓的获取等。各模块功能描述如表1.1所示：

表1.1 系统方案中各模块功能：
| 序号 | 子系统    | 功能描述                                                                   |
|----|--------|------------------------------------------------------------------------|
| 1  | 图像输入   | 调用MindX SDK的appsrc插件对视频数据进行拉流                                          |
| 2  | 图像放缩   | 调用MindX SDK的mxpi_imageresize                                           |
| 3  | 文本区域检测  | 通过文本区域检测模型，获取图像中的文本位置、以及像素点关联等数据                                           |
| 4  | 结果输出   | 输出文本位置以及像素点关联等信息    |
| 5  | 后处理   | 根据输出结果进行后处理，将结果写入到txt文件中    |
| 6  | 可视化   | 将txt文件中的文本区域坐标连线绘制到图像中    |


### 1.4 代码目录结构与说明

本工程名称为PixelLink，工程目录如下图所示：

```
.
├── model
│   └── convert.cfg  // 模型转化文件
├── pipeline
│   └── Pixel.pipeline
├── get_version.py
├── main.py
├── README.md
├── main_get_groundtruth.py
├── rrc_evaluation_funcs_1_1.py
└── script.py
```


### 1.5 技术实现流程图

![Pixellink文本检测流程图](https://images.gitee.com/uploads/images/2021/1029/112024_3a19c293_9366121.png "屏幕截图.png")



## 2 环境依赖

推荐系统为ubuntu 18.04，环境依赖软件和版本如下表：

| 软件名称  | 版本   |
| -------- | ------ |
| cmake    | 3.5+   |
| mxVision | 2.0.2  |
| python   | 3.7.5  |
| CANN     | 3.3.0  |
| Polygon3 | 3.0.9.1|


注意：在运行评测代码时，可能出现没有Polygon包的报错，可以pip install Polygon3导入。

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

pb模型提供在链接：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PixelLink/pixellink_tf.pb;

转换离线模型参考昇腾Gitee：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html

首先需要配置ATC环境，下载pb模型，放到相应的路径后，修改模型转换的cfg配置文件，配置文件已经上传至项目目录model下。使用命令

```
atc --model=pixellink_tf.pb --framework=3 --output=pixellink --output_type=FP32 --soc_version=Ascend310 --input_shape="Placeholder:1,768,1280,3"
--insert_op_conf=convert.cfg --log=info
```
转化项目模型。


注意：转化时，可根据需要修改输出的模型名称。注意模型以及转化所需文件的路径，防止atc命令找不到相关文件。


## 4 编译与运行
**步骤1**
下载项目文件，以及icdar2015数据集，其中项目文件链接在模型转换部分已经给出。数据集链接：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PixelLink/data.zip

zip文件中下载数据集ch4_test_image。

**步骤2**
  在安装mxVision SDK后，配置SDK安装路径、lib路径以及python路径，这些路径需要根据用户实际情况配置，例如SDK安装路径需要与用户本身安装路径一致，不一致将导致环境错误。同理，lib路径与python路径，都需要与实际情况一致。将下载的模型文件以及其他配置文件放到项目路径中，与pipeline内路径对应。修改pipeline内路径与模型文件一致。需要按照代码中的路径去创建文件路径，也可以根据实际需要修改代码中的路径变量。
  

**步骤3** 
与main.py同路径下创建ch4_test_images文件夹，将数据集解压后放到该文件夹内，可以从中取出一张图像，命名为test.jpg，并放到与main.py同路径下。

**步骤4**
运行推理代码：

```
python3.7 main.py
```
输出结果：可以直接得到这张测试图像的推理结果，该结果会存到一个txt文件中，并在同目录下可视化test.jpg的检测结果。可视化结果命名为my_test.jpg。

运行评测代码：

  将解压后的icdar2015数据集中的测试集部分解压到ch4_test_images文件夹中。ch4_test_images与main_get_groundtruth.py同目录，人工在main_get_groundtruth.py同目录下创建test文件夹，运行main_get_groundtruth.py，会生成数据集中每张图像的检测结果，检测结果会存放到./test/image_txt/目标路径下。需要人工将结果压缩为zip文件，命名为om_result.zip（可以根据需要命名为其他名称，但是后续运行评测代码时需要名称对应），压缩后将zip文件和groundtruth的zip文件（gt.zip）放到script.py路径下，gt.zip是原模型的groundtruth，用以作为评测的基准。该zip文件可以在链接：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PixelLink/data.zip 获取。最后，运行script.py，得到评测结果。运行评测代码文件路径要求如下图所示：

```
.
├── model
│   └── convert.cfg  // 模型转化文件
├── pipeline
│   └── Pixel.pipeline
├── get_version.py
├── main.py
├── README.md
├── main_get_groundtruth.py
├── rrc_evaluation_funcs_1_1.py
├── ch4_test_images
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ......(other images)
├── gt.zip
├── om_result.zip（人工运行main_get_groundtruth.py后生成的txt文件压缩为zip，命名为om_result.zip）
├── test
│   └── image_txt
└── script.py
```



```
python3.7 main_get_groundtruth.py
python3.7 script.py -g=./gt.zip -s=./om_result.zip
```
输出结果：首先得到本模型在ICDAR2015测试集上的推理结果，结果生成在./test/image_txt/下。再通过运行脚本代码可以得到原模型输出结果与本模型的结果的对比，最后得到本模型的平均指标。

