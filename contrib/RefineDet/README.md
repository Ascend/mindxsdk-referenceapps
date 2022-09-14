## RefineDet目标检测

## 1介绍

RefineDet基于MindXSDK开发，在昇腾芯片上进行目标检测，并实现可视化呈现。输入单张图片，对其进行推理，输出推理结果。

### 1.1 支持的产品

本产品以昇腾310（推理）卡为硬件平台。

### 1.2 支持的版本

该项目支持的SDK版本为2.0.4，CANN版本为5.0.4。

### 1.3 软件方案介绍

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统         | 功能描述                                                     |
| ---- | -------------- | ------------------------------------------------------------ |
| 1    | 图片输入       | 接收外部调用接口的输入视频路径，对视频进行拉流，并将拉去的裸流存储到缓冲区（buffer）中，并发送到下游插件。 |
| 2    | 图片解码       | 用于解码，将jpg格式图片解码为YUV                             |
| 3    | 数据分发       | 对单个输入数据进行2次分发。                                  |
| 4    | 数据缓存       | 输出时为后续处理过程创建一个线程，用于将输入数据与输出数据解耦，并创建缓存队列，存储尚未输入到下流插件的数据。 |
| 5    | 图像处理       | 对解码后的YUV格式的图像进行放缩。                            |
| 6    | 模型推理插件   | 目标检测。                                                   |
| 7    | 模型后处理插件 | 对模型输出的张量进行后处理，得到物体类型数据。               |
| 8    | 目标框转绘插件 | 物体类型转化为OSD实例                                        |
| 9    | OSD可视化插件  | 实现物体可视化绘制。                                         |
| 10   | 图片编码插件   | 用于将OSD可视化插件输出的图片进行编码，输出jpg格式图片。     |



### 1.4 代码目录结构与说明

本项目名为RefineDet目标检测，项目目录如下所示：

````
.
├── build.sh
├── config
│   ├── RefineDet.aippconfig
│   └── refine_det.cfg
├── main.py
├── models
│   ├── RefineDet.om
│   ├── RefineDet320_VOC_final_no_nms.onnx
│   └── VOC.names
├── eval
│   ├── myeval.py
│   └── precision_analysis/		# 精度验证步骤2中下载
│   	└── VOC/				# 精度验证步骤1中下载
├── out.jpg
├── pipeline
│   ├── refinedet.pipeline
│   └── fortest.pipeline
├── plugin
│   └── RefineDetPostProcess
│       ├── build.sh
│       ├── CMakeLists.txt
│       ├── RefineDetPostProcess.cpp
│       └── RefineDetPostProcess.h
└── test.jpg
````



### 1.5 技术实现流程图

![流程图](images/process.png)



### 1.6 特性及适用场景

本项目根据VOC数据集训练得到，适用于对以下类型物体的目标检测，并且将位置、物体类型、置信度标出。

````
飞机、自行车、鸟、船、瓶子、公交车、汽车、猫、椅子、牛、餐桌、狗、马、摩托车、人、大象、羊、沙发、火车、电视
````






## 2 环境依赖

推荐系统为ubuntu  18.04,环境软件和版本如下：

| 软件名称            | 版本  | 说明                          | 获取方式                                                  |
| ------------------- | ----- | ----------------------------- | :-------------------------------------------------------- |
| MindX SDK           | 2.0.4 | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk)       |
| ubuntu              | 18.04 | 操作系统                      | 请上ubuntu官网获取                                        |
| Ascend-CANN-toolkit | 5.0.4 | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial) |



在编译运行项目前，需要设置环境变量：

MindSDK 环境变量：

```
. ${SDK-path}/set_env.sh
```

CANN 环境变量：

```
. ${ascend-toolkit-path}/set_env.sh
```

- 环境变量介绍

```
SDK-path: mxVision SDK 安装路径
ascend-toolkit-path: CANN 安装路径
```




## 3 软件依赖说明

本项目无特定软件依赖。



## 4 模型转化

本项目中使用的模型是RefineDet模型，onnx模型可以直接[下载](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/47d31ca99aa641b2b220cabc9233cdb7)。下载后解包，得到`RefineDet320_VOC_final_no_nms.onnx`，使用模型转换工具ATC将onnx模型转换为om模型，模型转换工具相关介绍参考[链接](https://support.huawei.com/enterprise/zh/doc/EDOC1100234054?idPath=23710424|251366513|22892968|251168373)

模型转换步骤如下：

1、按照2环境依赖设置环境变量

2、`cd`到`models`文件夹，运行

````
atc --framework=5 --model=RefineDet320_VOC_final_no_nms.onnx --output=RefineDet --input_format=NCHW --input_shape="image:1,3,320,320" --log=debug --soc_version=Ascend310 --insert_op_conf=../config/RefineDet.aippconfig --precision_mode=force_fp32
````

3、执行该命令后会在指定输出.om模型路径生成项目指定模型文件RefineDet.om`。若模型转换成功则输出：

```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```

aipp文件配置如下：

```
aipp_op {
    related_input_rank : 0
    src_image_size_w : 320
    src_image_size_h : 320
    crop : false
    aipp_mode: static
    input_format : YUV420SP_U8
    csc_switch : true
    rbuv_swap_switch : true
    matrix_r0c0 : 256
    matrix_r0c1 : 454
    matrix_r0c2 : 0
    matrix_r1c0 : 256
    matrix_r1c1 : -88
    matrix_r1c2 : -183
    matrix_r2c0 : 256
    matrix_r2c1 : 0
    matrix_r2c2 : 359
    input_bias_0 : 0
    input_bias_1 : 128
    input_bias_2 : 128
    mean_chn_0 : 104
    mean_chn_1 : 117
    mean_chn_2 : 123
    min_chn_0 : 0.0
    min_chn_1 : 0.0
    min_chn_2 : 0.0
    var_reci_chn_0 : 1.0
    var_reci_chn_1 : 1.0
    var_reci_chn_2 : 1.0
}
```



## 5 编译运行

`main.py`：用来生成单张图片推理的可视化结果，以提供推理模型的应用实例。

1、`cd`到`plugin/RefineDetPostProcess`目录，运行`bash build.sh`，编译后处理插件

2、`config/refine_det.cfg` 确认权限`640`。

3、准备好测试图片`test.jpg`，放置在项目根目录。

4、在主目录，运行`python3 main.py`

最后会得到`out.jpg`即为输出结果



## 6 精度验证

本模型使用VOC数据集进行训练与精度评估。

1、运行以下命令，进行下载与解压。

````
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
````

2、根据[精度评估工具](https://gitee.com/mikochi13/mindxsdk-referenceapps/tree/master/tools/precision_analysis)进行安装与使用。上述两个下载的文件，按照代码目录结构中放置。

3、修改代码

将`./precision_analysis/utils/parser.py`的47行从`bbox = [x0, y0, w, h]`改为`bbox = [x0, y0, x1, y1]`

将`./precision_analysis/utils/parser.py`的86行从`    return class_id`改为`    return k`

将`./precision_analysis/interface/eval.py`的56-57行间插入

````
        with open("./result.json", 'w') as f:
            f.write(str(ret))
        print("get result.json successfully")
        exit()
````

4、将`voc2007val.json`移动到`eval/precision_analysis/VOC/VOCdevkit`目录。这里`voc2007val.json`为`VOC`数据集转`COCO`数据集后的格式文件，具体参考[教程](https://blog.csdn.net/sinat_28371057/article/details/114683354)。这里直接提供`voc2007val.json`文件。

5、在`precision_analysis`目录运行

````
python main.py --mode test.ssd_mobilenet_fpn.evaluation -data-loading-path ${VOC数据集路径} -label-loading-path ${VOC数据集标签路径} -pipeline-cfg-path ${SDK_pipeline文件路径} -stream-name ${pipeline配置stream名称}
````

配置 ${VOC数据集路径}：配置VOC数据集图片路径，例如：`./VOC/VOCdevkit/VOC2007/JPEGImages`

配置 ${VOC数据集标签路径}：配置数据集标签文件路径，例如：`./VOC/VOCdevkit/voc2007val.json`

配置 ${SDK_pipeline文件路径}：运行的pipeline的存放路径，例如：`../../pipeline/fortest.pipeline`

配置 ${pipeline配置stream名称}：运行的pipeline中的stream名称，例如：`RefineDet`

这里我们给出参考指令：

````
python3 main.py --mode test.ssd_mobilenet_fpn.evaluation -data-loading-path ./VOC/VOCdevkit/VOC2007/JPEGImages -label-loading-path ./VOC/VOCdevkit/voc2007val.json -pipeline-cfg-path ../../pipeline/fortest.pipeline -stream-name RefineDet
````

运行结束后，会在该目录下生成`result.json`，即为正常执行结束。

6、在`eval`目录运行`python3 myeval.py`，等待一段时间后即可得到结果。

