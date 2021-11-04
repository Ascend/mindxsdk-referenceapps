# 安全帽识别

## 1 介绍

安全帽作为工作中一样重要的安全防护用品，主要保护头部，防高空物体坠落，防物体打击、碰撞。通过识别每个人是否戴上安全帽，可以对没戴安全帽的人做出告警。本项目支持2路视频实时分析，其主要流程为:分两路接收外部调用接口的输入视频路径，将视频输入。通过视频解码将264格式视频解码为YUV格式图片。模型推理使用YOLOv5进行安全帽识别，识别结果经过后处理完成NMS得到识别框。对重复检测出的没戴安全帽的对象进行去重。最后将识别结果输出为两路，并对没佩戴安全帽的情况告警。

### 1.1 支持的产品

昇腾310(推理)

### 1.2 支持的版本

本样例配套的CANN版本为[3.3.0](https://www.hiascend.com/software/cann/commercial)，MindX SDK版本为[2.0.2](https://www.hiascend.com/software/mindx-sdk/mxvision)。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)



### 1.3 代码目录结构与说明

本sample工程名称为HelmetIdentification，工程目录如下图所示：

```
├── Models
  ├── aipp_YOLOv5.config        # 模型转换配置文件
  ├──atc-env.sh                 # 模型转换脚本
  ├──YOLOv5_s.om           #推理模型om文件
  ├──YOLOv5_s.onnx         #推理模型onnx文件
  ├──Helmet_yolov5.cfg  #后处理配置文件
  ├──HelmetDetection.pipline # 安全帽识别推理流程pipline
  ├──imgclass.names  # 模型所有可识别类
  ├──main-env.sh   # 环境变量设置脚本
  ├──main.py         # 推理运行程序
  ├──modify_yolov5s_slice.py  #slice算子修改脚本
  ├──dy_resize.py  # resize算子修改
  ├──utils.py  # 数据处理及可视化脚本
├── plugins  
  ├──MxpiSelectedFrame # 跳帧插件
├── Test  
  ├──performance_test_main.py # 性能测试脚本  
  ├──select.py # 测试集筛选脚本  
  ├──parse_voc.py # 测试数据集解析脚本  
  ├──testmain.py # 测试主程序  
  ├──map_calculate.py # 精度计算程序
├── build.sh    
```



### 1.5 技术实现流程图

<img src="https://gitee.com/liu-kai6334/mindxsdk-referenceapps/raw/master/contrib/HelmetIdentification/image/image4.jpg" alt="image4" style="zoom: 80%;" />



## 2 环境依赖

环境依赖软件和版本如下表：

| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 2.0.2       | mxVision软件包                | [链接](https://www.hiascend.com/software/mindx-sdk/mxvision) |
| Ascend-CANN-toolkit | 3.3.0     | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | Ubuntu 18.04 | 操作系统                      | Ubuntu官网获取                                               |
| opencv-python       | 4.5.2.54     | 用于识别结果画框              | python3.7 -m pip install opencv-python                       |



在运行脚本main.py前（2.2章节），需要通过环境配置脚本main-env.sh设置环境变量,运行命令：

```shell
source main-env.sh
```
- 环境变量介绍

```bash
export MX_SDK_HOME=${MX_SDK_HOME}
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/arm64-linux/atc/ccec_compiler/bin:${install_path}/arm64-linux/atc/bin:${install_path}/atc/bin
export PYTHONPATH=/usr/local/python3.7.5/bin:${MX_SDK_HOME}/python
export ${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${install_path}/acllib/lib64:/usr/local/Ascend/driver/lib64:${MX_SDK_HOME}/include:${MX_SDK_HOME}/python

export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
export ASCEND_OPP_PATH=${install_path}/opp
export GST_DEBUG=3
```

注：其中SDK安装路径${MX_SDK_HOME}替换为用户的SDK安装路径;install_path替换为开发套件包所在路径。LD_LIBRARY_PATH用以加载开发套件包中llib库。GST_DEBUG用以mxpi_rtspsrc取流地址配置不正确时出现warning日志提示。



## 3.推理

#### 步骤1 模型转换

##### 1.1模型与软件依赖

 所用模型与软件依赖如下表所示。

| 软件名称                | 版本     | 获取方式                                                     |
| ----------------------- | -------- | ------------------------------------------------------------ |
| pytorch                 | 1.5.1    | [pytorch官网](https://pytorch.org/get-started/previous-versions/) |
| ONNX                    | 1.7.0    | pip install onnx==1.7.0                                      |
| helmet_head_person_s.pt | v2.0     | [原项目链接](https://github.com/PeterH0323/Smart_Construction)(选择项目中yolov5s权重文件，权重文件保存在README所述网盘中) |
| YOLOv5_s.onnx           | YOLOv5_s | [链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/HelmetIdentification/model.zip) |



##### 1.2 pt文件转换为onnx文件

1. 可直接获取已经转换好的YOLOv5_s.onnx文件，链接如1.1所示。此模型已经完成优化，不再使用dy_resize.py、modify_yolov5s_slice.py进行优化。可直接转换为om模型。

2. 若尝试pt文件转换为onnx文件，可获取[原项目](https://github.com/PeterH0323/Smart_Construction)代码，下载至本地。安装环境依赖**requirements.txt**在原项目中已给出（原项目使用pytorch 1.5.1框架），pt文件转换为onnx文件所需第三方库**ONNX**如1.1中方式安装。

3. 通过上述1.1中链接获取模型文件helmet_head_person_s.pt，下载到本地后保存至原项目weights文件中。使用原项目中的export.py将pt文件转换为onnx格式文件。运行：

```shell
python3.7 ./models/export.py --weights ./weights/helmet_head_person_s.pt --img 640 --batch 1
```

其中onnx算子版本为opset_version=11。转换完成后权重文件helmet_head_person_s.onnx改名为YOLOv5_s.onnx上传至服务器任意目录下。



##### 1.3 onnx文件转换为om文件

  转换完成后onnx脚本上传至服务器任意目录后先进行优化。

1. 利用附件脚本dy_resize.py修改模型resize算子。该模型含有动态Resize算子（上采样），通过计算维度变化，改为静态算子，不影响模型的精度，运行如下命令：

```shell
python3.7 modify_yolov5s_slice.py YOLOv5_s.onnx
```

2. 然后利用modify_yolov5s_slice.py脚本修改模型slice算子，运行如下命令：

```bash
python3.7 modify_yolov5s_slice.py YOLOv5_s.onnx
```

可以得到修改好后的YOLOv5_s.onnx模型

3. 最后运行atc-env脚本将onnx转为om模型，运行命令如下：

```shell
sh atc-env.sh
```

提示 **ATC run success** 说明转换成功

脚本中包含atc命令:

```shell
--model=${Home}/YOLOv5_s.onnx --framework=5 --output=${Home}/YOLOv5_s  --insert_op_conf=./aipp_YOLOv5.config --input_format=NCHW --log=info --soc_version=Ascend310 --input_shape="images:1,3,640,640"
```

其参数如下表所示

| 参数名           | 参数描述                                                     |
| ---------------- | :----------------------------------------------------------- |
| --  framework    | 原始框架类型。当取值为5时，即为ONNX网络模型，仅支持ai.onnx算子域中opset v11版本的算 子。用户也可以将其他opset版本的算子（比如opset v9），通过PyTorch转换成 opset v11版本的onnx算子 |
| --model          | 原始模型文件路径与文件名                                     |
| --output         | 如果是开源框架的网络模型，存放转换后的离线模型的路径以及文件名。 |
| --soc_version    | 模型转换时指定芯片版本。昇腾AI处理器的版本，可从ATC工具安装路径的“/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/atc/data/platform_config”目录下 查看。 ".ini"文件的文件名即为对应的${soc_version} |
| --insert_op_conf | 插入算子的配置文件路径与文件名， 例如aipp预处理算子。        |
| --input_shape    | 模型输入数据的 shape。                                       |
| --out_nodes      | 指定输出节点,如果不指定输出节点（算子名称），则模型的输出默认为最后一层的算子信息，如果 指定，则以指定的为准 |

其中--insert_op_conf参数为aipp预处理算子配置文件路径。该配置文件aipp_YOLOv5.config在输入图像进入模型前进行预处理。该配置文件保存在源码Models目录下。

注：1. [aipp配置文件教程链接](https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0015.html)
   2.atc-env.sh脚本内 Home 为onnx文件所在路径。


#### 步骤2 模型推理 

##### 2.1 pipline编写

pipline根据1.5节中技术实现流程图编写，该文件**HelmetDetection.pipline**放在源码根目录Models。

 注： 

1.pipline中mxpi_modelinfer用于加载yolov5安全帽识别模型。该插件包含四个参数，modelPath用于加载om模型文件。labelPath用于加载模型可识别类（imgclass.names）。postProcessLibPath用于加载后处理动态链接库文件，该模块实现NMS等后处理。postProcessConfigPath用于加载后处理所需要的配置文件（Helmet_yolov5.cfg）。本项目使用后处理文件为**libMpYOLOv5PostProcessor.so**。该后处理配置文件内容如下：              

```python
CLASS_NUM=3
BIASES_NUM=18
BIASES=10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326
SCORE_THRESH=0.4
OBJECTNESS_THRESH=0.3
IOU_THRESH=0.5
YOLO_TYPE=3
ANCHOR_DIM=3
MODEL_TYPE=1
RESIZE_FLAG=0
```

注：pipline中以上四个参数要修改为相应文件所在绝对路径。

2.pipline中mxpi_selectedframe插件完成视频跳帧。对于输入帧率为24fps输入视频进行每三帧抽一帧进行识别。实现8fps的帧率。

将目录切换至./plugins/MxpiSelectedFrame

输入如下命令编译生成mxpi_selectedframe.so：

```shell
mkdir build
cd build
cmake ..
make -j
```

编译成功后将产生**libmxpi_selectedframe.so**文件，文件生成位置在build目录下。将其复制至SDK的插件库中(./MindX_SDK/mxVision/lib/plugins)

 注：[插件编译生成教程](https://support.huaweicloud.com/mindxsdk201/index.html)在《用户手册》深入开发章节

3.pipline中涉及到的**绝对路径**都要修改成用户安装sdk文件相应的路径

##### 2.2 运行推理

编写完pipline文件后即可运行推理流程进行识别，该程序**main.py**放在源码根目录Models。

mian.py通过调用sdk接口创建多个流完成数据接收、处理以及输出，接口调用流程图如下所示：

<img src="https://gitee.com/liu-kai6334/mindxsdk-referenceapps/raw/master/contrib/HelmetIdentification/image/image1.jpg" alt="image1" style="zoom:80%;" />

本项目通过mxpi_rtspsrc拉流输入数据，通过两路GetResult接口输出数据，一路输出带有帧信息的图片数据，一路输出带有帧信息的目标检测框和检测框跟踪信息。推理过程如下：

首先通过[live555](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/Live555离线视频转RTSP说明文档.md)进行推流，进入到live555安装目录下mediaServer路径，上传要推流的视频在本目录下然后推流。 live555只支持特定几种格式文件，不支持MP4。 所以本地文件先要转成live555支持的格式。选择使用[ffmpeg](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/pc端ffmpeg安装教程.md)进行格式转换。

转换命令如下：

```shell
ffmpeg -i xxx1.mp4 -vcodec h264 -bf 0 -g 25 -r 24 -s 1280*720 -an -f h264 xxx2.264
```

注：参数如下：

| 参数    | 作用                                                   |
| ------- | ------------------------------------------------------ |
| -i      | 表示输入的音视频路径需要转换视频                       |
| -f      | 强迫采用特定格式输出                                   |
| -r      | 指定帧率输出                                           |
| -an     | 关闭音频                                               |
| -s      | 分辨率控制                                             |
| -g      | 关键帧间隔控制                                         |
| -vcodec | 设定视频编解码器，未设定时则使用与输入流相同的编解码器 |

转换完成后上传视频至live555安装目录下mediaServer。输入命令进行推流：

```shell
./live555MediaServer test.264
```

test.264可替换成任意上传至当前目录的[264格式文件](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/pc端ffmpeg安装教程.md)，如要修改相应的也要在pipline中修改mxpi_rtspsrc的拉流路径

![image2](https://gitee.com/liu-kai6334/mindxsdk-referenceapps/raw/master/contrib/HelmetIdentification/image/image2.jpg)

然后切换目录至main.py所在目录下，运行命令：

```shell
python3.7.5 main.py
```

即可得到输出结果，输出结果将原来的两路视频分为两个文件保存，utils.py中的oringe_imgfile用于设置图像输出路径,用户需手动建立输出文件output，文件路径可自定义设置。本项目文件放置规范如下：

![image3](https://gitee.com/liu-kai6334/mindxsdk-referenceapps/raw/master/contrib/HelmetIdentification/image/image3.jpg)

所有数据放置于output中，one 、two为两路视频输出文件。

输出结果有如下几种情况：
| 序号 | 输入                           | 输出                                                       |
| ---- | ------------------------------ | ---------------------------------------------------------- |
| 1    | 两路只有一路输入               | 只打印有输入一路的输出                                     |
| 2    | 无输入或输入视频中无可识别对象 | 打印：Object detection  result of model infer is null!!!   |
| 3    | 输入视频有识别对象             | 打印每次推理的head的帧信息的尺寸与识别结果                       |
| 4    | 识别对象未佩戴安全帽           | 打印：Warning:Not  wearing a helmet, InferenceId：FrameId: |

#### 步骤3 测试性能与精度


##### 3.1 性能测试

性能测试使用脚本performance_test_main.py，该脚本与main.py大体相同，不同之处是在performance_test_main.py中添加了时间戳测试，测试数据为mxpi_rtspsrc拉取的视频流。两路视频尺寸分别取多组不同尺寸的视频做对比。推理三百帧图片后取平均时间值，设置如下环境变量：

```shell
export PYTHONPATH=/usr/local/python3.7.5/bin:${MX_SDK_HOME}/python:{path}
```

注：{path}设置为根目录中Models所在路径

运行如下命令得到结果：

```shell
python3.7 performance_test_main.py
```

注：1.与运行main.py时相同，运行performance_test_main.py时要先使用live555进行推流。**测试视频**上传至[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/HelmetIdentification/test_video.zip)，该视频为不同尺寸不同帧率的同一视频。如test64036830_158s.264为尺寸640×640，帧率30，时长158s的视频。

2.performance_test_main.py中加载pipline文件应写HelmetDetection.pipline的绝对路径


#####  3.2 精度测试

###### 3.2.1 数据集说明

- 数据集来源:  [Safety-Helmet-Wearing-Dataset](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/HelmetIdentification/data.zip)

- 数据集结构

  ```
  ├── VOC2028
    ├── Annotations                 # 图片解释文件，含标签等信息，与JPEGImages中图片一一对应
    ├── ImageSets                 # 存放txt文件
    ├── JPEGImages                 # 数据集原图片
  ```

注：将数据集中的三个文件放置于项目的根目录Test文件下，与**select.py**同目录。

###### 3.2.2测试数据集筛选

依据数据集中ImageSets文件夹中test.txt文件，从原始数据集中筛选出测试数据集，该程序**select.py**放在源码根目录Test中，在同目录下创建文件夹TestImages用来存储筛选的数据。在该目录下运行命令：
```shell
python3.7.5 select.py
```

程序运行后在根目录Test中会存放筛选出的测试集图片共1517张。

###### 3.2.3 测试数据集解析

解析测试数据集，在同级目录下生成类别文件**voc.names**、图片信息文件**VOC2028.info**和真实标签文件夹**ground_truth**， 该程序**parse_voc.py**放在源码根目录Test中。

运行命令：

```shell
python3.7.5 parse_voc.py 
```

###### 3.2.4 推理运行

依据编写的pipline业务流，对测试数据集进行推理，输出结果保存在同级目录**detection-test-result**文件夹中，该文件需要手动建立。程序**testmain.py**文件放在源码根目录Test中。

注：输出推理结果文件txt中数据格式为：

```shell
cls conf x0 y0 x1 y1
```

其中cls表示识别区域所属类别，conf表示识别置信度，(x0,y0)表示识别区域左上角点的坐标，(x1,y1)表示识别区域右下角点的坐标。  

运行命令：

```shell
python3.7.5 testmain.py
```

注：testmain.py中直接写入了pipline，其中mxpi_modelinfer插件四个参数的配置与HelmetDetection.pipline完全相同。

###### 3.2.5 精度计算

推理完成后，依据图片真实标签和推理结果，计算精度。输出结果保存在同级目录**output**文件夹中，该文件需要手动建立。程序map_calculate.py文件放在源码根目录Test中。

注：测试数据集中图片有两类标签"person"(负样本，未佩戴安全帽)和"hat"(正样本，佩戴安全帽)。模型输出标签有三类"person"、"head"、"helmet"，其中"head"与真实标签"person"对应，"helmet"与真实标签"hat"对应。在**map_calculate.py**文件中做了对应转换处理。  

运行命令：

```shell
python3.7.5 map_calculate.py --label_path  ./ground-truth  --npu_txt_path ./detection-test-result/ -na -np
```

即可得到输出。其中precision、recall和map记录在**output/output.txt**文件中。



## 5 软件依赖说明

推理中涉及到第三方软件依赖如下表所示。

| 依赖软件 | 版本                      | 说明                           |
| -------- | ------------------------- | ------------------------------ |
| live555  | 1.09                      | 实现视频转rstp进行推流         |
| ffmpeg   | 2021-08-08-git-ac0408522a | 实现mp4格式视频转为264格式视频 |

注：1.[live555使用教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/Live555离线视频转RTSP说明文档.md)

​        2.[ffmpeg使用教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/pc端ffmpeg安装教程.md)

## 6 常见问题

### 6.1 图片格式问题

**问题描述：**

E0628 10:14:48.309166 8155  DvppImageDecoder.cpp:152] [mxpi_imagedecoder0] [2006] [DVPP:ecode jpeg or jpg fail]

**解决方案：**

本项目只支持jpg图片输入 如输入其他格式会报如上错误

