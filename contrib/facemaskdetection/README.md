# 口罩识别参考设计

## 1 介绍
识别图片中的人是否佩戴口罩。图片数据经过 抽帧、解码后，送给口罩检测模型推理。

### 1.1 支持的产品

本项目以昇腾Atlas 500 A2 / Atlas 200I DK A2为主要的硬件平台。

### 1.2 支持的版本

支持的SDK版本为5.0.RC1。

CANN版本为6.2.RC1。

### 1.3 代码目录结构与说明

本sample工程名称为口罩识别参考设计，工程目录如下图所示：

```
.
├── README.md                     # 模型转换配置文件
├── anchor_decode.py              # 计算bbox
├── anchor_generator.py           # 生成先验框
├── image.py											# 图片识别主程序
├── main.pipeline									# 口罩识别推理流程pipline
├── models												# 推理模型文件夹
│   └── face_mask.aippconfig			# 转模型前处理配置文件
├── nms.py												# nms计算程序
├── test_image.py									# 精度测试程序
├── map_calculate.py							# mAP计算程序
└── xmltotxt.py										# 数据机标定xml转txt程序
```



### 1.5 技术实现流程图



<img src="./image/image1.png" alt="image2" style="zoom:50%;" />



## 2 环境依赖

环境依赖软件和版本如下表：

| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 5.0.RC1        | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| Ascend-CANN-toolkit | 6.2.RC1     | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | ubuntu 22.04 | 操作系统                      | Ubuntu官网获取                                               |
| opencv-python       | 4.5.2.54     | 用于识别结果画框              | python3 -m pip install opencv-python                       |


在编译运行项目前，需要执行如下两个环境配置脚本设置环境变量：

```shell
. /usr/local/Ascend/ascend-toolkit/set_env.sh   # Ascend-cann-toolkit开发套件包默认安装路径，根据实际安装路径修改
. ${MX_SDK_HOME}/mxVision/set_env.sh   # ${MX_SDK_HOME}替换为用户的SDK安装路径
```

## 依赖下载

所用模型与软件依赖如下表所示。

| 软件名称               | 版本  | 获取方式                                                     |      |      |      |
| ---------------------- | ----- | ------------------------------------------------------------ | ---- | ---- | ---- |
| face_mask_detection.pb | SSD   | [GitHub](https://github.com/AIZOOTech/FaceMaskDetection/blob/master/models/face_mask_detection.pb) |      |      |      |
| benchmark工具          | 1.0.0 | [arm](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/tool/benchmark.aarch64) [x86](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/tool/benchmark.x86_64)|      |      |      |
| 后处理开源部分 | python   | [GitHub](https://github.com/AIZOOTech/FaceMaskDetection/tree/master/utils) |      |      |      |

- 开源代码部分的部署
>下载开源代码中utils文件夹内的3个py文件(anchor_decode.py,anchor_generator.py,nms.py)并放置于项目根目录即可，最终的目录结构参见 [1.3 代码目录结构与说明]

##  推理

#### 步骤1 模型转换

pb文件转换为om文件，若使用A200I DK A2运行，推荐使用PC转换模型，具体方法可参考A200I DK A2资料。

1. 执行如下脚本设置环境变量：

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh   # Ascend-cann-toolkit开发套件包默认安装路径，根据实际安装路径修改
```

2. 运行atc工具将pb模型文件转为om模型，运行命令如下。

```
atc --model=./face_mask_detection.pb --framework=3 --output=./aipp --output_type=FP32 --soc_version=Ascend310B1 --input_shape="data_1:1,260,260,3" --input_format=NHWC --insert_op_conf=./face_mask.aippconfig
```

提示 **ATC run success** 说明转换成功

####  步骤2 模型推理

##### pipline编写

pipline根据1.5节中技术实现流程图编写，该文件**main.pipeline**放在源码根目录。

##### 运行推理

编写完pipline文件后即可运行推理流程进行识别，该程序**image.py**放在源码根目录。

可在image.py中134行修改进行推理的原图地址。![image2](./image/image2.png)

在根目录下，运行命令：

```
python3.9.2 image.py
```

即可得到输出结果，输出结果对原图像的目标以及口罩进行识别画框并将结果保存至根目录下**my_result.jpg**
