# Multi Thread

## 1.介绍

多线程调用SDK的发送和接收数据接口函数

## 2.环境依赖

- 支持的硬件形态和操作系统版本

| 硬件形态                             | 操作系统版本   |
| ------------------------------------ | -------------- |
| x86_64+Atlas 300I 推理卡（型号3010） | Ubuntu 18.04.1 |
| x86_64+Atlas 300I 推理卡（型号3010） | CentOS 7.6     |
| ARM+Atlas 300I 推理卡 （型号3000）   | Ubuntu 18.04.1 |
| ARM+Atlas 300I 推理卡 （型号3000）   | CentOS 7.6     |

- 软件依赖

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.5.1+ |
| Python   | 3.9.2  |
| opencv-python   | 3.4+  |
| mmcv   |  -  |

支持的SDK版本为 5.0.RC1, CANN 版本310使用6.3.RC1，310B使用6.2.RC1。

## 3.预准备

脚本转换为unix格式

```bash
sed -i 's/\r$//' ./*.sh
```
给script目录下的脚本添加执行权限

配置环境变量

```bash
export MX_SDK_HOME=${安装路径}/mxVision
```

## 4 模型转换

yolov3模型下载参考华为昇腾社区[ModelZoo](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ActionRecognition/ATC%20YOLOv3%28FP16%29%20from%20TensorFlow%20-%20Ascend310.zip)  
使用以下命令进行转换，请注意aipp配置文件名，此处使用的为自带sample中的相关文件（{Mind_SDK安装路径}/mxVision/samples/mxVision/models/yolov3/）

```
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"

# 说明：若用例执行在310B上，则--soc_version=Ascend310需修改为Ascend310B1
```


## 5.运行

### 5.1 运行前配置
- 在`MultiThread`下创建`models`目录
```bash
mkdir models
```
- 获取pipeline文件中推理所需的om等文件，并拷贝到`models`路径下

- 将测试图片（仅支持.jpg格式），拷贝到`MultiThread`下的 `picture`目录（必须包含test.jpg）

- 修改run.sh文件的32行，选择执行的py文件（不同的py文件中调用不同的SDK函数或者使用不同的调用方式）

### 5.2 运行

- 使用默认pipeline文件

```bash
bash run.sh
```
注意：

1.调用main.py文件时，对应EasyStream.pipeline文件(此样例只获取picture/test.jpg)

2.调用main_sendprotobuf.py文件时，对应EasyStream_protobuf.pipeline文件(此样例获取picture文件下的所有.jpg)

- 自定义pipeline文件（例如：mypipeline.pipeline)

```bash
bash run.sh ./mypipeline.pipeline
```

### 5.3 结果

main.py运行结果格式如下：

`End to get data, threadId = 1, result = b'{"MxpiObject":[{"classVec":[{"classId":15,"className":"cat","confidence":0.98471146799999998,"headerVec":[]}],"x0":86.182579000000004,"x1":247.33078,"y0":86.406199999999998,"y1":442.07312000000002},{"classVec":[{"classId":16,"className":"dog","confidence":0.99579948200000001,"headerVec":[]}],"x0":220.453766,"x1":434.736786,"y0":132.42176799999999,"y1":466.86648600000001}]}'`

main_sendprotobuf.py运行结果格式如下：

`result:  cat:0.9853515625
 result:  dog:0.978515625
 result:  dog:0.99658203125
 result:  dog:0.99609375
 result:  dog:0.99560546875
 result:  dog:0.9951171875
 poss stream process finish
`

## 6.FAQ

### 6.1 运行程序时,cv2报错 

参照`AllObjectsStructuring`样例的README.md文件中的`4 准备`的步骤8安装相关依赖库

### 6.2 运行程序时，mmcv报错

执行`pip3.9 install mmcv`下载相关依赖库