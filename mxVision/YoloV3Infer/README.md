# 基于C++ V2接口的yoloV3推理

## 1 简介

基于C++ V2接口的yoloV3推理样例使用mxVision SDK进行开发，以昇腾Atlas310P/310B卡为主要的硬件平台，主要支持以下功能：

1. 图片读取解码：本样例支持JPG及PNG格式图片，使用图像处理单元进行解码。
2. 图片缩放/保存：使用图像处理单元相关接口进行图片的缩放，并输出一份缩放后的副本保存。
3. 模型推理：使用yoloV3网络识别输入图片中对应的目标，并打印输出大小。
4. 模型后处理：使用SDK中的模型后处理插件对推理结果进行计算，并输出相关结果，

## 2 环境依赖

- 支持的硬件形态和操作系统版本

| 硬件形态                             | 操作系统版本   |
| ----------------------------------- | -------------- |
| x86_64+Atlas 310P 推理卡（型号3010） | Ubuntu 18.04.1 |
| ARM+Atlas 310P 推理卡 （型号3000）   | Ubuntu 18.04.1 |
| ARM+Atlas 310B 推理卡 （型号A500 A2）   | openEuler |

- 软件依赖

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.10+ |
| mxVision | 5.0RC3    |
| Python   | 3.9.2+  |



## 3 代码主要目录介绍

本代码仓名称为mxSdkReferenceApps，工程目录如下图所示：

```
|-- YOLOV3CPPV2
|   |-- CMakeLists.txt
|   |-- main.cpp
|   |-- README.md
|   |-- run.sh
|   |-- test.jpg    #测试使用的图片，需要用户自备
|   |-- model
|   |   |-- yolov3_tf_bs1_fp16.cfg
|   |   |-- aipp_yolov3_416_416.aippconfig
|   |   |-- yolov3_tf_bs1_fp16.OM       #OM模型需要自行下载转换
|   |   |-- yolov3.names

```
## 4 模型获取

**步骤1** 在ModelZoo上下载YOLOv3模型。[下载地址](https://gitee.com/link?target=https%3A%2F%2Fobs-9be7.obs.cn-east-2.myhuaweicloud.com%2F003_Atc_Models%2Fmodelzoo%2Fyolov3_tf.pb)

**步骤2** 将获取到的YOLOv3模型pb文件存放至`./model/`。

**步骤3** 模型转换

在`./model/`目录下执行一下命令

```bash
# 设置CANN环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 执行，转换YOLOv3模型
# Execute, transform YOLOv3 model. 310P

atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310P3 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"

# Execute, transform YOLOv3 model. 310B

atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310B2 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"
# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
```

执行完模型转换脚本后，会生成相应的.om模型文件。 执行后终端输出为：
```bash
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```


## 5 准备

**步骤1：** MindX SDK安装前准备可参考《用户指南》，[安装教程](https://www.hiascend.com/document/detail/zh/mind-sdk/50rc2/vision/mxvisionug/mxvisionug_0006.html)

**步骤2：** 配置 mxVision SDK 环境变量。

`bash ${安装路径}/set_env.sh `

注：mxVision SDK安装路径一般建议使用 /home/{$username}/MindX_SDK。

**步骤3：** 准备模型，根据本文档上一章节转换样例需要的OM模型,并将模型保存到./model/目录下。

**步骤4：** 修改配置根目录下的配置文件./CMakeLists.txt文件和run.sh脚本：

1. 将所有“MX_SDK_HOME”字段中未配置的值替换为实际使用安装路径

2. 检查Cmkaelist.txt中对应的include_directories和link_directories是否正确，包含使用相对目录的SDK路径和绝对路径的CANN路径

3. 检查run.sh中的路径和文件是否正确，包含使用相对目录的SDK路径和需要推理的图片名。

**步骤5：** 准备测试图片，在根目录下放置待测试图片，并修改run.sh中的文件名。

## 5 运行

- 简易运行样例
> `bash run.sh`
- 完成运行样例
> 完成编译后，以测试图片作为参数运行生成的mxbaseV2_sample即可

## 6 其他
对于需要自行开发后处理的用户，请修改YoloV3PostProcess函数完成对应功能