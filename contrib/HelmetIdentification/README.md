# 安全帽识别

## 1 介绍
安全帽作为工作中一样重要的安全防护用品，主要保护头部，防高空物体坠落，防物体打击、碰撞。通过识别每个人是否戴上安全帽，可以对没戴安全帽的人做出告警。本项目支持2路视频实时分析，其主要流程为:分两路接收外部调用接口的输入视频路径，将视频输入。通过视频解码将264格式视频解码为YUV格式图片。通过跳帧实现每路8fps输出。模型推理使用YOLOv5进行安全帽识别，识别结果经过后处理完成NMS得到识别框。对重复检测出的没戴安全帽的对象进行去重。最后将识别结果输出为两路，并对没佩戴安全帽的情况告警。

### 1.1 支持的产品

昇腾310(推理)

### 1.2 支持的版本

本样例配套的CANN版本为[3.2.0](https://www.hiascend.com/software/cann/commercial)，MindX SDK版本为[21.0.1](https://www.hiascend.com/software/mindx-sdk/mxvision)。

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
├── plugins  
  ├──MxpiSelectedFrame # 跳帧插件
├── Test  
  ├──select.py # 测试集筛选脚本  
  ├──parse_voc.py # 测试数据集解析脚本  
  ├──testmain.py # 测试主程序  
  ├──map_calculate.py # 精度计算程序
├── build.sh    
```



### 1.5 技术实现流程图

<img src="https://gitee.com/liu-kai6334/mindxsdk-referenceapps/raw/master/contrib/HelmetIdentification/image/image4.jpg" alt="image4" style="zoom:80%;" />



## 2 环境依赖

环境依赖软件和版本如下表：

| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 21.0.1       | mxVision软件包                | [链接](https://www.hiascend.com/software/mindx-sdk/mxvision) |
| Ascend-CANN-toolkit | 20.2.rc1     | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | Ubuntu 18.04 | 操作系统                      | Ubuntu官网获取                                               |
| opencv-python       | 4.5.2.54     | 用于识别结果画框              | python3.7 -m pip install opencv-python                       |



在运行脚本main.py前（2.2章节），需要通过环境配置脚本main-env.sh设置环境变量：

- 环境变量介绍

```bash
export MX_SDK_HOME=${MX_SDK_HOME}
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/arm64-linux/atc/ccec_compiler/bin:${install_path}/arm64-linux/atc/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:/usr/local/python3.7.5/bin:/usr/local/lib/python3.7/dist-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${install_path}/acllib/lib64:/usr/local/Ascend/driver/lib64:${install_path}/arm64-linux/atc/lib64:${install_path}/acllib_linux.arm64/lib64:${MX_SDK_HOME}/include:${MX_SDK_HOME}/python${LD_LIBRARY_PATH}

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

| 软件名称                | 版本  | 获取方式                                                     |
| ----------------------- | ----- | ------------------------------------------------------------ |
| pytorch                 | 1.5.1 | [pytorch官网](https://pytorch.org/get-started/previous-versions/) |
| ONNX                    | 1.7.0 | pip install onnx==1.7.0                                      |
| helmet_head_person_s.pt | v2.0  | [原项目链接](https://github.com/PeterH0323/Smart_Construction)(选择项目中yolov5s权重文件，权重文件保存在README所述网盘中) |
| YOLOv5_s.onnx           | YOLOv5_s | [链接](https://pan.baidu.com/s/15qjahlaO9TTd6orzGrm0Sw) 提取码：b123 |



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
python3.7 modify_yolov5s_slice.py yolov5_s.onnx
```

2. 然后利用modify_yolov5s_slice.py脚本修改模型slice算子，运行如下命令：

```bash
python3.7 modify_yolov5s_slice.py yolov5_s.onnx
```

可以得到修改好后的yolov5_s.onnx模型

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

其中--insert_op_conf参数为aipp预处理算子配置文件路径。该配置文件在输入图像进入模型前进行预处理。该配置文件保存在源码Models目录下,其内容如下：

```python
aipp_op{
    aipp_mode:static
    input_format : YUV420SP_U8
    src_image_size_w : 640
    src_image_size_h : 640
    crop: false
    load_start_pos_h : 0
    load_start_pos_w : 0
    crop_size_w : 640
    crop_size_h: 640
    csc_switch : true
    rbuv_swap_switch : false
    # 色域转换
    matrix_r0c0: 256
    matrix_r0c1: 0
    matrix_r0c2: 359
    matrix_r1c0: 256
    matrix_r1c1: -88
    matrix_r1c2: -183
    matrix_r2c0: 256
    matrix_r2c1: 454
    matrix_r2c2: 0
    input_bias_0: 0
    input_bias_1: 128
    input_bias_2: 128
    # 均值归一化
    min_chn_0 : 0
    min_chn_1 : 0
    min_chn_2 : 0
    var_reci_chn_0: 0.003921568627451
    var_reci_chn_1: 0.003921568627451
    var_reci_chn_2: 0.003921568627451}
```

注：1. [aipp配置文件教程链接](https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0015.html)
   2.atc-env.sh脚本内 ${Home} 为onnx文件所在路径。


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

![image1](https://gitee.com/liu-kai6334/mindxsdk-referenceapps/raw/master/contrib/HelmetIdentification/image/image1.png)

本项目通过mxpi_rtspsrc拉流输入数据，通过两路GetprotoBuf接口输出数据，一路输出带有帧信息的图片数据，一路输出带有帧信息的目标检测框和检测框跟踪信息。推理过程如下：

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

即可得到输出结果，输出结果将原来的两路视频分为两个文件保存，oringe_imgfile用于设置图像输出路径，infer_imgfile用于设置告警图片输出路径。用户可自定义设置任意文件路径。本项目文件放置规范如下：

![image3](https://gitee.com/liu-kai6334/mindxsdk-referenceapps/raw/master/contrib/HelmetIdentification/image/image3.jpg)

所有数据放置于output中，one 、two为两路视频输出文件。image用于存放模型识别后图片。inference用于存放识别出的未佩戴安全帽目标所在帧，每个目标只输出一次。

#### 步骤3 测试性能与精度

##### 3.1 性能测试

下载[benchmark工具包](https://support.huawei.com/enterprise/zh/software/251707127-ESW2000346827)解压至服务器任意文件夹。切换到工具包所在目录，使用root权限运行以下命令得到推理结果：              

```shell
chmod +x benchmark.aarch64

./benchmark.aarch64 -model_type=yolocaffe -batch_size=1 -device_id=0 -om_path=./YOLOv5_s.om -input_width=640 -input_height=640 -input_imgFiles_path=./test_imgFiles -useDvpp=true -output_binary=False 
```

参数说明具体如下表所示：

| 参数                       | 参数说明                                                     |
| -------------------------- | ------------------------------------------------------------ |
| -model_type                | 模型的类型，当前支持如下几种：● vision：图像处理        ● nmt：翻译        ● widedeep：搜索         ● yolocaffe：Yolo目标检测          ● nlp：自然语言处理          ● bert：语义理解 |
| -batch_size                | 执行一次模型推理所处理的数据量                               |
| -device_id                 | 运行的Device编号，请根据实际使用的Device修 改。缺省值为0。   |
| -om_path                   | 经过ATC转换后的模型OM文件所在的路径                          |
| -input_imgFiles_path       | 模型对应的图片文件夹所在的路径。                             |
| -input_height/-input_width | 输入图片尺寸                                                 |
| -useDvpp                   | 模型前处理是否使用DVPP 编解码模块。取值为： ● true  ● false   缺省值为false。若使用 DVPP编解码或图像缩放， 则该参数置为true。 |
| -output_binary             | 输出结果格式是否为二进制文件（即bin文件）。取值为 true：输出结果格式为 bin文件。 |

注：

1.[benchmark推理使用指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100191895/a3eb1384)

2.[性能测试所用数据集下载](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)



#####  3.2 精度测试

###### 3.2.1 数据集说明

- 数据集来源:  [Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)

- 数据集结构

```
├── Test
  ├── Annotations                 # 图片解释文件，含标签等信息，与JPEGImages中图片一一对应
  ├── ImageSets                 # 存放txt文件
  ├── JPEGImages                 # 数据集原图片
```

- 测试数据集筛选
  依据ImageSets文件夹中test.txt文件，从原始数据集中筛选出测试数据集，该程序**select.py**放在源码根目录Test中，其内容如下：

```python
import os
import shutil
import cv2

with open("ImageSets/Main/test.txt", "r") as f:
    data = f.readlines()
    text_data = []
    for line in data:
        line_new = line.strip('\n')  # 去掉列表中每一个元素的换行符
        text_data.append(line_new)
    print(text_data)

path = 'JPEGImages'
save_path = 'TestImages'

for file in os.listdir(path):
    file_name = file.split('.')[0]
    if file_name in text_data:
        img = cv2.imread(path + '/' + file)
        cv2.imwrite(save_path + '/' + file_name + ".jpg",img)
```

程序运行后在根目录Test中生成TestImages文件夹，存放筛选出的测试集图片，共1517张。

运行命令：

```shell
python3.7.5 select.py
```

###### 3.2.2 测试数据集解析

解析测试数据集，在同级目录下生成类别文件**voc.names**、图片信息文件**VOC2028.info**和真实标签文件夹**ground_truth**， 该程序**parse_voc.py**放在源码根目录Test中。

运行命令：

```shell
python3.7.5 parse_voc.py 
```

###### 3.2.3 推理运行

依据编写的pipline业务流，对测试数据集进行推理，输出结果保存在同级目录**detection-test-result**文件夹中，该程序**testmain.py**文件放在源码根目录Test中。

注：输出推理结果文件txt中数据格式为：

```shell
cls conf x0 y0 x1 y1
```

其中cls表示识别区域所属类别，conf表示识别置信度，(x0,y0)表示识别区域左上角点的坐标，(x1,y1)表示识别区域右下角点的坐标。  

运行命令：

```shell
python3.7.5 testmain.py
```

###### 3.2.4 精度计算

推理完成后，依据图片真实标签和推理结果，计算精度。输出结果保存在同级目录**output**文件夹中，该程序map_calculate.py文件放在源码根目录Test中。

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

