# 无人机遥感旋转目标检测

## 1 介绍
本项目基于MindXSDK开发，针对无人机遥感图像进行目标检测。遥感图像是俯瞰拍摄获得，其包含的空间场景大且复杂，包含的目标种类和数量要比一般的检测任务多。从检测框形式上划分，目标检测任务可以分为水平检测和旋转检测，本项目属于后者，输入一张待检测图片，可以输出目标旋转角度检测框，并有可视化呈现。

遥感图像分辨率高，包含的小目标数量多且密集分布，如果直接将整幅图像进行推理检测，检测效果无法满足预期。所以在将待检测图片输入到模型推理之前，需要先对其进行裁剪操作，本方案是将高分的遥感图像裁剪为多张固定分辨率（1024×1024）的图片，如果待检测图片不足1024×1024分辨率，则将图片等比例放大、补边后得到所需分辨率的图片。为了保证裁剪图片边缘信息被保留，裁剪时会在图片与图片之间设定200像素的重叠区域。

本方案是采用YOLOv5变体模型，将裁剪的图片输入到模型进行推理，推理得到128×128，64×64，32×32三个尺度的特征图，这些特征数据中包含预测边界框的位置信息，类别概率信息以及旋转角度概率信息，从中筛选出符合要求的边界框信息并进行去重操作，然后将这些框信息绘制在输入图像上，完成单张裁剪图片的检测。所有裁剪图片检测完成之后，进行图像的融合，由于图像裁剪过程中设有重叠区域，在图像拼接之前需再进行去重操作，最后将所有框信息标注到遥感图像上，完成整张图片的检测。

### 1.1 支持的产品

本项目以昇腾Atlas310卡为主要硬件平台。

### 1.2 支持的版本

支持的SDK版本为2.0.2。

### 1.3 软件方案介绍

单张裁剪得到的图片进行旋转目标检测由SDK业务流来完成，检测业务流程为：待检测图片通过 appsrc 插件输入，然后使用图像解码插件mxpi_imagedecoder对图片进行解码，再通过图像缩放插件mxpi_imageresize将图像缩放至满足检测模型要求的输入图像大小要求，缩放后的图像输入模型推理插件mxpi_tensorinfer得到检测结果，本项目开发的rotateobjpostprocess旋转目标检测插件用来处理推理结果，获取目标检测框的中心点坐标、尺寸、旋转角度、类别Id、置信度等信息，最后通过输出插件appsink获取旋转目标检测的输出结果，在pipeline外部通过调用opencv库进行检测框的可视化绘制。

表1 系统方案中各模块功能：

| 序号 | 模块名       | 功能描述                               |
| ---- | :----------- | -------------------------------------- |
| 1    | 图片输入     | 获取 jpg 格式输入图片                  |
| 2    | 图片解码     | 解码图片                               |
| 3    | 图片缩放     | 将输入图片放缩到模型指定输入的尺寸大小 |
| 4    | 模型推理     | 对输入张量进行推理                     |
| 5    | 旋转目标检测 | 从模型推理结果中获取检测框信息         |
| 6    | 结果输出     | 将检测框信息输出                       |
| 7    | 结果可视化   | 将检测得的框信息绘制在输入图片上       |



### 1.4 代码目录结构与说明

本工程名称为RotateObjectDetection，工程目录如下图所示：

```
.
├── build.sh
├── patch
├── Readme_image
│   ├── flow chart.png
│   ├── model structure.png
│   ├── overlapping area.png
│   ├── OverallProcess.jpg
│   └── polygon area.png
├── plugins
│   ├── build.sh
│   ├── CMakeLists.txt
│   ├── MxpiRotateObjPostProcess.cpp
│   └── MxpiRotateObjPostProcess.h
├── proto
│   ├── build.sh
│   ├── CMakeLists.txt
│   └── mxpiRotateobjProto.proto
├── python
│   ├── main.py
│   ├── models
│   │   ├── model_conversion_pt2onnx.py
│   │   ├── aipp_yolov5_1024_1024.aippconfig
│   │   └── model_conversion.sh
│   └── pipeline
│       └── RotateObjectDetection.pipeline
└── README.md
```





### 1.5 技术实现流程图

#### 1.5.1 项目整体流程

本项目整体流程如图1所示。

![整体流程图](https://gitee.com/xiang-cui/mindxsdk-referenceapps/raw/master/contrib/RotateObjectDetection/Readme_image/flow%20chart.png)

​                                                                                                         图1 无人机遥感旋转目标检测流程图

#### 1.5.2 旋转目标检测后处理插件实现流程

 本项目开发的旋转目标检测插件的输入是YOLOv5变体模型输出，模型从输入到输出的映射由图2所示。模型输入为固定分辨率1024×1024的图片，输出形状大小分别为1×3×128×128×201、1×3×64×64×201、1×3×32×32×201的特征矩阵，其中3为先验框anchor的数量；128×128、64×64、32×32为特征图的宽高，是YOLO网络在进行目标检测时划分的grid数量；201为网络预测一个bounding box所需的参数数量，参数所代表的具体含义如图2中单个预测目标所示。`tx, ty, tl, ts`为预测目标相对于anchor的偏移量，o为objectness值，模型训练所用`DOTAv1.5`数据集包含16个类别，`c1,c2,...,c16`对应16个类别的概率得分，模型在进行角度预测时，将该问题作为分类问题，`p1,p2,...,p180`为1到180度的角度概率。

<img src="https://gitee.com/xiang-cui/mindxsdk-referenceapps/raw/master/contrib/RotateObjectDetection/Readme_image/model%20structure.png" alt="模型输入输出映射" style="zoom:50%;" />

​                                                                                                     图2 YOLOv5模型输入输出映射图

本项目开发的rotateobjpostprocess后处理插件对模型输出数据的处理步骤如下：

(1) **通过索引获取对应的参数信息，计算获得检测框的中心点坐标、尺寸、类别Id、旋转角度等信息**。模型输出采用长边表示法`(x_c , y_c, longside, shortside, θ)`，需要将其转换为opencv中的 minAreaRect格式`（x_c, y_c, width, height, θ)`，计算并保存检测框四个角点的坐标用于计算两个旋转框的IOU值。

(2) **设置confidence阈值为0.25，剔除掉confidence小于该阈值的检测框**。

(3) **设置IOU阈值为0.4，通过非极大值抑制删除重叠框**。

旋转检测框与水平检测框的IOU计算方式不同。两个水平检测框的重叠区域为矩形，如图3中第一幅图所示，只需要比较两个检测框的左上和右下两个角点坐标就可得到重叠矩形的角点坐标，再通过坐标相减得到重叠矩形的尺寸即可计算重叠面积。而两个旋转框的重叠区域是不规则的多边形，图3中第2、3、4幅图仅展示了重叠区域为三角形、四边形、五边形的情况，计算两个旋转框的IOU值则需要计算多边形面积。

<img src="https://gitee.com/xiang-cui/mindxsdk-referenceapps/raw/master/contrib/RotateObjectDetection/Readme_image/overlapping%20area.png" alt="重叠区域示意图" style="zoom: 80%;" />

​                                                                                                                  图3 检测框重叠区域示意图

通过n边形角点坐标计算面积的方法：选取一个起始角点，逆时针遍历多变形的 *n* 个角点，计算相邻两个角点与原点构成三角形的有向面积，将所有三角形的有向面积累加起来就可得到多边形面积。示例如下，在图4中选取点 *c* 为起始角点，四边形`ABCD`的面积计算如下。
$$
S=S_ΔOCB+S_ΔOBA+ S_ΔOAD+ S_ΔODC
$$
上式中*S*为有向三角形面积，三角形`OCB`、`OBA`、`OAD`为正三角形，`ODC`为负三角形，将四个三角形面积相加即可得到四边形`ABCD`的面积。

<img src="https://gitee.com/xiang-cui/mindxsdk-referenceapps/raw/master/contrib/RotateObjectDetection/Readme_image/polygon%20area.png" alt="多边形面积" style="zoom:50%;" />

​                                                                                                             图4 计算多边形面积示意图

两个旋转框重叠区域的角点可以通过计算两个检测框各边的交点获得。计算得到两个旋转框的重叠面积后，并集面积通过两个矩形面积相加减去重叠面积获得，两者相除，即可得到旋转框的IOU值。

(4) **设定自定义proto结构体，对结构体赋值并发送到metadata中，可供pipeline外部获取数据并进行画图操作**。



## 2 环境依赖

推荐系统为ubantu 18.04，环境依赖软件和版本如下表：

| 软件名称      | 版本          |
| ------------- | :------------ |
| cann          | 1.77.22.6.220 |
| cmake         | 3.5+          |
| mxVision      | 2.0.2         |
| python        | 3.7.5         |
| opencv-python | 4.5.3         |
| swig          | 4.0.2         |

在编译运行项目前，需要设置环境变量：

```shell
export MX_SDK_HOME={SDK安装路径}/mxVision
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${MX_SDK_HOME}/python:${PYTHONPATH}
# swig环境变量请参照依赖安装1
# export PATH={swig安装路径}/bin:$PATH
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
```

- 环境变量介绍

```
MX_SDK_HOME: mxVision SDK 安装路径
LD_LIBRARY_PATH: lib库路径
PYTHONPATH: python环境路径
```

### 依赖安装

1.本项目中用到`DOTA`数据集的开发工具`DOTA_devkit`，由数据集作者开源发布，可以实现对`DOTA`数据集的图片裁剪，结果融合，精度测试等功能。

`DOTA_devkit`开发工具的开源链接：https://github.com/CAPTAIN-WHU/DOTA_devkit

由于工具默认处理`png`格式的图片，融合结果的脚本缺少画图功能，且测试精度的脚本默认已经存在保存图片名称的文件和检测结果按照类别分类的文件，所以本项目用的脚本在原项目的基础上做了部分修改，修改内容保存在**项目目录**下的patch文件中，请根据以下指令对文件进行修改。

请先fork一个原作者的代码仓，并且clone到本地，请执行以下指令，命令中：用户名为github的账户名

```bash
git clone https://github.com/{用户名}/DOTA_devkit.git
git branch patch
git checkout patch
```

将**项目目录**下的patch文件拷贝到克隆的`DOTA_devkit`目录下，执行以下指令

```bash
git apply patch
git commit -a -m "Patch Apply"
```

将修改好的代码包放在项目目录下，项目目录结构如下

```
.
├── build.sh
├── patch
├── DOTA_devkit
│   ├── dota-v1.5_evaluation_task1.py
│   ├── SplitOnlyImage.py
│   ├── ResultMerge.py
│   ├── polyiou.cpp
│   └── other files
├── Readme_image
│   ├── flow chart.png
│   ├── model structure.png
│   ├── overlapping area.png
│   ├── OverallProcess.jpg
│   └── polygon area.png
├── plugins
│   ├── build.sh
│   ├── CMakeLists.txt
│   ├── MxpiRotateObjPostProcess.cpp
│   └── MxpiRotateObjPostProcess.h
├── proto
│   ├── build.sh
│   ├── CMakeLists.txt
│   └── mxpiRotateobjProto.proto
├── python
│   ├── main.py
│   ├── models
│   │   ├── model_conversion_pt2onnx.py
│   │   ├── aipp_yolov5_1024_1024.aippconfig
│   │   └── model_conversion.sh
│   └── pipeline
│       └── RotateObjectDetection.pipeline
└── README.md
```

在工具使用之前，需要进行以下环境安装：

步骤1. 安装swig工具，若无权限，请离线安装swig工具，并为其添加环境变量

```shell
sudo apt-get install swig
```

离线安装，环境变量添加指令：

```shell
export PATH={swig安装路径}/bin:$PATH
```

步骤2. 为python创建C++扩展

在**项目路径**下执行以下指令：

```bash
cd DOTA_devkit
swig -c++ -python polyiou.i
python3.7 setup.py build_ext --inplace
```

以上指令成功执行后，会在当前路径下生成`polyiou.py`，`polyiou_wrap.cxx`，`_polyiou.cpython-37m-aarch64-linux-gnu.so`等文件。

2.安装opencv-python，执行以下指令

```shell
python3.7 -m pip install opencv-python
```
3.安装shapely，执行以下指令

```bash
apt-get install libgeos-dev
python3.7 -m pip install shapely
```

## 3 模型转换

###  3.1 参考模型

本项目中使用的模型是YOLOv5变体模型，参考实现代码：https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB

pytorch模型下载链接：https://mindx.sdk.obs.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/RotateObjectDetection/model.zip

###  3.2 pt模型转onnx模型

注意：由于模型结构中归一化层使用SyncBatchNorm层，onnx模型导出只适用于GPU设备。如果您没有GPU设备，请跳过此步骤，直接下载onnx模型。

1.从3.1中提供的参考实现代码链接中下载参考项目文件，得到`YOLOv5_DOTA_OBB-master`文件夹。

2.按照参考项目`YOLOv5_DOTA_OBB-master`文件夹中`requirement.txt`文件配置pytorch环境。

3.从3.1中提供的pytorch模型下载链接中下载模型权重文件`YOLOv5_DOTAv1.5_OBB.pt`,放置在`YOLOv5_DOTA_OBB-master/weights`目录下，将本项目目录下的`python/models/model_convert_pt2onnx.py`同样放置到`YOLOv5_DOTA_OBB-master/weights`目录下。

4.在`YOLOv5_DOTA_OBB-master/weights`目录下运行命令

```
python model_convert_pt2onnx.py
```

执行成功后会在当前目录下生成转换得到的onnx模型，默认文件名为`YOLOv5_DOTAv1.5_OBB_1024_1024.onnx`

转换生成的onnx模型下载链接：https://mindx.sdk.obs.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/RotateObjectDetection/model.zip

###  3.3 onnx模型转om模型

将3.2中转换得到的onnx模型权重文件拷贝到本项目目录下的`python/models`目录下，将其转换为om模型，转化步骤如下：

* 进入`python/models`目录下

* 执行命令

  ```bash
  bash model_conversion.sh
  ```
  注意：如果shell脚本执行时报错：`line: XXX "$'\r': command not found" `，请执行以下指令修改脚本格式，没有报错请忽略此处。

  ```bash
  dos2unix model_conversion.sh
  ```

  `model_conversion.sh`脚本中包含atc命令

  ```bash
  atc --model=./YOLOv5_DOTAv1.5_OBB_1024_1024.onnx --framework=5 --output=./YOLOv5_DOTAv1.5_OBB_1024_1024 --input_format=NCHW --log=info --soc_version=Ascend310 --insert_op_conf=./aipp_yolov5_1024_1024.aippconfig --input_shape="images:1,3,1024,1024"
  ```

  命令中`--model`属性为原始模型路径以及文件名，`--output`属性为转换后的om模型的存放路径以及文件名，`--insert_op_conf`属性为`aipp`预处理算子配置文件，检测图片输入到模型推理之前需要进行resize、crop、通道变换、色域转换、均值归一化等预处理，具体操作请查看`python/models/`目录下的`aipp_yolov5_1024_1024.aippconfig`文件，`--input_shape`属性为模型输入图片的尺寸信息。

  执行上述命令后，终端输出为：

  ```shell
  ATC start working now, please wait for a moment.
  ATC run success, welcome to the next use.
  ```

  表示命令执行成功，在当前目录下生成`YOLOv5_DOTAv1.5_OBB_1024_1024.om`模型文件。

## 4 编译与运行

**步骤1**  按照第2小节**环境依赖**中的步骤设置环境变量。

**步骤2**  按照第3小节**模型转换**的步骤获得om模型文件。

**步骤3**  编译生成后处理动态链接库，并将其拷贝到MindX SDK插件库目录下。

在**项目目录**下执行命令

```bash
bash build.sh
```

注意：如果shell脚本执行时报错：`line: XXX "$'\r': command not found" `，**项目路径**下请执行以下指令修改脚本格式，没有报错请忽略此处。

```bash
dos2unix build.sh
cd ./proto
dos2unix build.sh
cd ../plugins
dos2unix build.sh
```

将生成的动态链接库拷贝到SDK插件目录下，**项目路径**下执行指令：

```bash
cp plugins/build/libmxpi_rotateobjpostprocess.so ${MX_SDK_HOME}/lib/plugins/
```

**步骤4**  图片检测

注意：本项目只支持jpg格式图片的检测

* **将待检测遥感图像进行裁剪**

  * 在**项目路径**下创建`image`文件夹，将待检测图像`test.jpg`放于该目录下。
  * 修改`DOTA_devkit/SplitOnlyImage.py`脚本文件中第69行的`splitbase`函数参数图片源路径为`'../image'`，裁剪后的目标路径为`'../imageSplit'`。

  ```
  split = splitbase(r'../image',
                    r'../imageSplit')
  ```

  * 在**项目路径**下执行以下命令：

  ```shell
  cd DOTA_devkit
  python3.7 SplitOnlyImage.py
  ```

  执行上述命令后，在终端显示：

  ```
  test split down！
  ```

  表示图像裁剪完成，项目路径下生成`imageSplit`文件夹，裁剪后的图片位于该文件夹中，目录如下

  ```
  .
  ├── image
  │   └── test.jpg
  ├── imageSplit
  │   ├── test_x__xxx___xxx.jpg
  │   ├── test_x__xxx___xxx.jpg
  │   └── 其它裁剪生成的图片
  └── 其它文件
  ```

  例如一张裁剪后的图片名为`test_1__0___824.jpg`，命名含义：`test`为图片原名，1为缩放比例，0，824表示以原图像横坐标为0，纵坐标为824的位置为起始点，裁剪一张1024×1024的图片，这些数据用于图片融合中的坐标映射。

* **对裁剪图片进行目标检测**

  * 在**项目路径**下执行以下命令（使用`cd ..`回到项目路径）

  ```bash
  cd python
  python3.7 main.py --input-path ../imageSplit --output-path ../detection
  ```

  终端输出如下：

  ```shell
  Detect the order of the image: 1
  detect_file_name: test__1__0___0.jpg
  I1113 10:40:31.478523 31706 MxpiRotateObjPostProcess.cpp:749] MxpiRotateObjPostProcess::Process start
  I1113 10:41:19.014143 31706 MxpiRotateObjPostProcess.cpp:831] MxpiRotateObjPostProcess::Process end
  Detect result:  484 small-vehicle, 21 large-vehicle
  Detection time: 48.35641884803772 s
  ...
  ```

  `Detect the order of the image` 表示当前检测图片位于所有待检测图片中的顺序， `detect_file_name`表示当前检测图片名称， `Detect result`为检测结果中各类别的数量， `Detection time`为检测当前图片花费的时间。  

  `main.py`脚本文件中默认输入路径为`../imageSplit` ，默认输出路径为`../detection`，请根据需要修改。

  命令执行完毕后，会在项目路径下生成`detection`文件夹，存放检测生成的图片以及存放目标检测框信息的txt文件，目录结构如下

  ```
  .
  ├── image
  ├── imageSplit
  ├── detection
  │   ├── result_txt
  │   │   └── result_before_merge
  │   │          └── test.txt
  │   ├── test_x__xxx___xxx.jpg
  │   └── 其它检测生成的图片
  └── 其它文件
  ```
  
  由于检测图片中小目标多密集分布，打印类别标签会导致检测信息遮挡，默认不打印labels，如需打印，请执行指令：
  
  ```shell
  python3.7 main.py --input-path ../imageSplit --output-path ../detection --labels_print
  ```
  
* **对检测结果进行融合**

  修改`DOTA_devkit/ResultMerge.py`脚本中269行，273行`mergebypoly`,`draw_merged_img`函数的路径为：

  ```python
  mergebypoly(r'../detection/result_txt/result_before_merge', 
              r'../detection/result_txt/result_merged')
  draw_merged_img(srcimgpath=r'../image', 
                  dstimgpath=r'../detection/merged_drawed',
                  mergedlabelpath=r'../detection/result_txt/result_merged',
                  labels=labels_print
                 )
  ```
  
  `mergebypoly`函数中`srcpath`：需要进行融合的文件的源路径，即未经融合的result文件，
                     `dstpath`：融合后结果存放的目标路径。
  `draw_merged_img`函数中，`imgsrcpath` ：原图片路径，
                          `dstpath` ：保存绘制图片的目标路径，
                          `mergedlabelpath `：融合结果存放路径。
  
  执行以下命令：
  
  ```shell
  cd ../DOTA_devkit
  python3.7 ResultMerge.py --draw_img
  ```
  
  终端输出如下：
  
  ```
  test merge down！
  test draw down！
  ```
  
  表示图片融合完成，在`detection`路径下会生成`merged_drawed`文件夹，其中存放融合后的图片。在`/detection/result_txt`路径下生成`result_merged`文件夹，存放融合后的目标检测框信息的txt文件。目录结构如下：
  
  ```
  .
  ├── image
  ├── imageSplit
  ├── detection
  │   ├── merged_drawed
  │   │	    └── test_merged.jpg
  │   └── result_txt
  │   │    ├── result_before_merge
  │   │    │   └── test.txt
  │   │    └── result_merged
  │   │        └── test.txt
  │   ├── test_x__xxx___xxx.jpg
  │   ├── test_x__xxx___xxx.jpg
  │	└── 其它检测生成的图片
  └──其它文件											
  ```

  如果希望融合后的图片显示标签信息，请执行指令：

  ```shell
  python3.7 ResultMerge.py --draw_img --labels_print
  ```

## 5 精度测试

1. 下载`DOTA`数据集，下载链接：https://mindx.sdk.obs.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/RotateObjectDetection/data.zip

2. 在项目目录下创建dataset文件夹，将数据集中的验证集部分`val_data`解压到`./dataset`目录下，确保下载完的数据集和标注文件后的项目目录为：

   ```
   .
   ├── dataset
   │   ├── images
   │   │   ├── P0003.jpg
   │   │   ├── P0004.jpg
   │   │   └── other images
   │   └── labelTxt
   │       ├── P0003.txt
   │       ├── P0004.txt
   │       └── other annotation files
   ├── build.sh
   ├── patch
   ├── DOTA_devkit
   │   ├── dota-v1.5_evaluation_task1.py
   │   ├── SplitOnlyImage.py
   │   ├── ResultMerge.py
   │   ├── polyiou.cpp
   │   └── other files
   ├── Readme_image
   │   ├── flow chart.png
   │   ├── model structure.png
   │   ├── overlapping area.png
   │   ├── OverallProcess.jpg
   │   └── polygon area.png
   ├── plugins
   │   ├── build.sh
   │   ├── CMakeLists.txt
   │   ├── MxpiRotateObjPostProcess.cpp
   │   └── MxpiRotateObjPostProcess.h
   ├── proto
   │   ├── build.sh
   │   ├── CMakeLists.txt
   │   └── mxpiRotateobjProto.proto
   ├── python
   │   ├── main.py
   │   ├── models
   │   │   ├── model_conversion_pt2onnx.py
   │   │   ├── aipp_yolov5_1024_1024.aippconfig
   │   │   └── model_conversion.sh
   │   └── pipeline
   │       └── RotateObjectDetection.pipeline
   └── README.md
   ```

3. 数据集切分

   修改`DOTA_devkit/SplitOnlyImage.py`文件中的第69行和70行为：

   ```python
   split = splitbase(r'../dataset/images',
                     r'../datasetSplit')
   ```

   **项目路径**下执行指令：

   ```shell
   cd DOTA_devkit
   python3.7 SplitOnlyImage.py
   ```

4. 对切分后的数据集进行检测，执行指令：

   ```shell
   cd ../python
   python3.7 main.py --input-path ../datasetSplit --output-path ../detection_evaluation
   ```

   验证集共458张图片，切分后生成5298张图片，检测完成大概需要1个小时50分钟

5. 对检测结果进行融合

   修改`DOTA_devkit/ResultMerge.py`中第269行`mergebypoly`函数的路径为：

   ```python
   mergebypoly(r'../detection_evaluation/result_txt/result_before_merge', 
               r'../detection_evaluation/result_txt/result_merged')
   ```

   执行指令：

   ```
   cd ../DOTA_devkit
   python3.7 ResultMerge.py
   ```

6. 对融合生成的结果进行精度测试，执行指令
   ```shell
   cd ../DOTA_devkit
   python3.7 dota-v1.5_evaluation_task1.py
   ```

   测试脚本中的默认路径在298，299行，如下所示

   ```python
   detection_output_path = r'../detection_evaluation'
   dataset_path = r'../dataset'
   ```

   `detection_output_path`：检测生成的文件路径

   `dataset_path`：数据集存放路径

   执行成功之后，终端输出每一个类别的ap以及总的map，输出信息如下：

   ```shell
   classname: plane
   ap:  0.9037683943017425
   classname: baseball-diamond
   ap:  0.669361186912927
   classname: bridge
   ap:  0.3707150833231114
   classname: ground-track-field
   ap:  0.45483809401631814
   classname: small-vehicle
   ap:  0.4823227112052795
   classname: large-vehicle
   ap:  0.6993688350158442
   classname: ship
   ap:  0.8945445019679292
   classname: tennis-court
   ap:  0.9070522668758366
   classname: basketball-court
   ap:  0.6092885386627254
   classname: storage-tank
   ap:  0.70005660820716
   classname: soccer-ball-field
   ap:  0.41082209453570584
   classname: roundabout
   ap:  0.5647113266429562
   classname: harbor
   ap:  0.6718642651267972
   classname: swimming-pool
   ap:  0.5911498492120622
   classname: helicopter
   ap:  0.5300178153550588
   classname: container-crane
   This class is not be detected in your dataset: container-crane
   map: 0.6306587714240969
   classaps:  [90.37683943 66.93611869 37.07150833 45.4838094  48.23227112 69.9368835
    89.4544502  90.70522669 60.92885387 70.00566082 41.08220945 56.47113266
    67.18642651 59.11498492 53.00178154]
   
   ```

   最终map检测结果为0.6307。

