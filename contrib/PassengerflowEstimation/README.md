# passengerflowestimation客流量检测

## 1介绍

passengerflowestimation基于MindXSDK开发，在昇腾芯片上进行客流量统计，将最后统计得到的客流量结果在终端内显示。输入一段视频，最后可以得出在某一时间内的客流量。

### 1.1支持的产品

本产品以昇腾310（推理）卡为硬件平台。

### 1.2支持的版本

该项目支持的SDK版本为2.0.4，CANN版本为5.0.4。

### 1.3软件方案介绍

基于MindX SDK的passengerflowestimation客流量检测业务流程为：待检测视频存放在live555服务器上经过mxpi_rtspsrc拉流插件输入，然后使用视频解码插件mxpi_videodecoder进行视频解码。用tee串流插件对解码的视频流进行分发。随后用mxpi_imageresize插件将图像方所至满足检测模型要求的输入图像大小要求，放缩后的图像输入模型推理插件mxpi_tensorinfer进行处理。将经过模型推理插件处理后的数据流输入到mxpi_objectpostprocessor之中，对目标检测模型推理的输出张量进行后处理。处理完毕后，再输入mxpi_selectobject插件中对人和车辆进行筛选，紧接着输入到mxpi_motsimplesortV2插件中实现多目标路径记录功能。本项目开发的mxpi_passengerflowestimation插件将统计识别到的人的数目并输出到终端(此处统计的人数是是通过规定直线人流量)。最后通过mxpi_videoencoder将输入的视频标记出被识别出的人后输出。

表1.1系统方案各子系统功能描述：

| 序号 | 子系统         | 功能描述                                                     |
| ---- | -------------- | ------------------------------------------------------------ |
| 1    | 视频输入流     | 接收外部调用接口的输入视频路径，对视频进行拉流，并将拉去的裸流存储到缓冲区（buffer）中，并发送到下游插件。 |
| 2    | 视频解码       | 用于视频解码，当前只支持H264格式。                           |
| 3    | 数据分发       | 对单个输入数据进行2次分发。                                  |
| 4    | 数据缓存       | 输出时为后续处理过程创建一个线程，用于将输入数据与输出数据解耦，并创建缓存队列，存储尚未输入到下流插件的数据。 |
| 5    | 图像处理       | 对解码后的YUV格式的图像进行放缩。                            |
| 6    | 模型推理插件   | 目标分类或检测。                                             |
| 7    | 模型后处理插件 | 对模型输出的张量进行后处理。                                 |
| 8    | 目标筛选插件   | 对需要进行统计的目标进行筛选。                               |
| 9    | 统计客流量插件 | 对视频中筛选出来的目标进行数量统计。                         |
| 10   | 目标框转绘插件 | 将流中传进的数据转换可用于OSD插件绘图所使用的MxpiOsdinstancesList数据类型。 |
| 11   | OSD可视化插件  | 实现对视频流的每一帧图像进行绘制。                           |
| 12   | 视频编码插件   | 用于将OSD可视化插件输出的图片进行视频编码，输出视频。        |

### 1.4代码目录结构与说明

本项目名为passengerflowestimation客流量检测，项目目录如下所示：

```
├── models
│   ├── aipp_Passengerflowdetection.config            # 模型转换aipp配置文件
│   ├── passengerflowestimation.onnx      # onnx模型
│   └── yolov4.om               # om模型
├── pipeline
│   └── passengerflowestimation.pipeline        # pipeline文件
├── plugins
│   ├── mxpi_passengerflowestimate     # passengerflowestimation后处理插件
│   │   ├── CMakeLists.txt        
│   │   ├── PassengerFlowEstimation.cpp  
│   │   ├── PassengerFlowEstimation.h
│   │   └── build.sh
│   └── mxpi_selectobject  # 筛选目标插件
│       ├── CMakeLists.txt
│       ├── mxpi_selectobject.cpp
│       ├── mxpi_selectobject.h
│       └── build.sh
├── CMakeLists.txt
├── build.sh
├── main.cpp
```

### 1.5技术实现流程图

本项目实现对输入视频的人流量进行统计，流程图如下：

![passengerdetect.png](https://s2.loli.net/2022/04/20/ea1387YxRJiFzrD.png)



## 2环境依赖

推荐系统为ubuntu  18.04,环境以来软件和版本如下：

| 软件名称            | 版本  | 说明                          | 获取方式                                                  |
| ------------------- | ----- | ----------------------------- | --------------------------------------------------------- |
| MindX SDK           | 2.0.4 | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk)       |
| ubuntu              | 18.04 | 操作系统                      | 请上ubuntu官网获取                                        |
| Ascend-CANN-toolkit | 5.0.4 | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial) |

在项目编译运行时候，需要设置环境变量：

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
. ${SDK安装路径}/mxVision/set_env.sh

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

注：其中SDK安装路径${MX_SDK_HOME}替换为用户的SDK安装路径。并且本项目用到了mxpi_opencvosd插件，使用mxpi_opencvosd插件前，需要使用osd相关的模型文件，请执行MindX SDK开发套件包安装目录下operators/opencvosd/generate_osd_om.sh脚本生成所需模型文件（在generate_osd_om.sh所在文件夹下执行`bash generate_osd_om.sh `，若这条命令执行失败，则将passengerflowestimation目录下的.om文件移动到generate_osd_om.sh所在的文件夹目录下MindXSDK安装路径/mxVision/operators/opencvosd下）。{install_path}替换为开发套件包所在路径。**（注：开头两行为每次一重新开启终端执行程序就需要输入，此外的其他为转换模型需要，若已经转换模型成功，则不需要输入这些）**



## 3 软件依赖

推理中涉及到第三方软件依赖如下表所示。

| 软件名称 | 版本       | 说明                           | 使用教程                                                     |
| -------- | ---------- | ------------------------------ | ------------------------------------------------------------ |
| live555  | 1.09       | 实现视频转rstp进行推流         | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md) |
| ffmpeg   | 2021-07-21 | 实现mp4格式视频转为264格式视频 | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/pc%E7%AB%AFffmpeg%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B.md#https://ffmpeg.org/download.html) |



## 4 模型转换

本项目中使用的模型是yolov4模型，onnx模型可以直接[下载](https://www.hiascend.com/zh/software/modelzoo/detail/1/abb7e641964c459398173248aa5353bc)。下载后使用模型转换工具ATC将onnx模型转换为om模型，模型转换工具相关介绍参考链接：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html

模型转换步骤如下：

1. 从链接处下载onnx模型至`passengerflowestimation/models`文件夹下，将模型修改名称为`passengerflowestimation.onnx`。
2. 进入`passengerflowestimation/models`文件夹下面执行命令**（注：提前设置好环境变量）**：

```
atc --model=${模型路径}/passengerflowestimation.onnx --framework=5 --output=${输出.om模型路径}/yolov4 --input_format=NCHW --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,3,608,608" --log=info --insert_op_conf=${aipp文件路径}/aipp_Passengerflowdetection.config 
```

执行该命令后会在指定输出.om模型路径生成项目指定模型文件`passengerflowestimation.om`。若模型转换成功则输出：

```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```

模型转换的aipp文件配置如下：

```
aipp_op{
    aipp_mode:static
    input_format : YUV420SP_U8

    src_image_size_w : 608
    src_image_size_h : 608

    crop: false
    load_start_pos_h : 0
    load_start_pos_w : 0
    crop_size_w : 608
    crop_size_h: 608

    csc_switch : true
    rbuv_swap_switch : true
    
    
    min_chn_0 : 0
    min_chn_1 : 0
    min_chn_2 : 0
    var_reci_chn_0: 0.003921568627451
    var_reci_chn_1: 0.003921568627451
    var_reci_chn_2: 0.003921568627451

    
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
    input_bias_2: 128}
```

## 5准备

按照第3小结**软件依赖**安装live555和ffmpeg，按照 [Live555离线视频转RTSP说明文档](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md)将mp4视频转换为h264格式。并将生成的264格式的视频上传到`live/mediaServer`目录下，然后修改`passengerflowestimation/pipeline`目录下的`passengerflowestimation.pipeline`文件中mxpi_rtspsrc0的内容。

```
"mxpi_rtspsrc0": {
            "factory": "mxpi_rtspsrc",
            "props": {
                "rtspUrl":"rtsp://xxx.xxx.xxx.xxx:xxxx/xxx.264",      // 修改为自己所使用的的服务器和文件名
                "channelId": "0"
            },
            "next": "mxpi_videodecoder0"
        },
```

## 6编译与运行

### 步骤1

按照第二小节环境依赖中的步骤设置环境变量。

### 步骤2

按照第四小节模型转换中的步骤获取om模型文件，放置在`passengerflowestimation/models`目录下。

### 步骤3 修改

对于mxpi_passengerflowestimate插件的使用说明：

在pipeline中，mxpi_passengerflowestimate插件如下：

```
"mxpi_passengerflowestimation0": {
                "props": {
                    "dataSource": "mxpi_selectobject0",
                    "motSource": "motV2",
                    "x0":"736",
                    "y0":"191",
                    "x1":"1870",
                    "y1":"191"
                },
                "factory": "mxpi_passengerflowestimation",
                "next": "mxpi_object2osdinstances0"
            },
```

这里点$(x_0,y_0)$与$(x_1,y_1)$确定了一条线段，这个插件统计经过该线段的客流量。

### 步骤4 编译

进入passengerflowestimation目录，在passengerflowestimation目录下执行命令：

```
bash build.sh
```

命令执行成功之后会在passengerflowestimation/plugins/mxpi_passengerflowestimation和passengerflowestimation/plugins/mxpi_selectobject目录下分别生成build文件夹。将build文件夹下生成的.so下载后上传到${SDK安装路径}/mxVision/lib/plugins目录下。在生成build文件夹后，进入到build目录下执行如下指令：

```
chmod 640 libmxpi_passengerflowestimation.so
chmod 640 libmxpi_selectobject.so
```



### 步骤5 运行：

在passengerflowestimation目录下运行：

```
python3 main.py
```

最后生成的结果将会在passengerflowestimation文件夹目录下result.h264文件里面。



## 7性能测试

在程序运行过程中，会自动输出帧率：

![image1.png](https://s2.loli.net/2022/04/22/m7DMpSjK8W4zfNk.png)

帧率已经达到要求。



