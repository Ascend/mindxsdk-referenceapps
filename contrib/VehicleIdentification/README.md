# 车型识别

## 1 介绍
在本系统中，目的是基于MindX SDK，在昇腾平台上，开发端到端车型识别的参考设计，实现对图像中的车辆进行车型识别的功能，并把可视化结果保存到本地，达到功能要求。

样例输入：带有车辆的jpg图片。

样例输出：框出并标有车辆车型与置信度的jpg图片。

### 1.1 支持的产品

本项目以昇腾Atlas310或Atlas310B卡为主要的硬件平台。本项目以昇腾Atlas310或Atlas310B卡为主要的硬件平台。本项目以昇腾Atlas310或Atlas310B卡为主要的硬件平台。


### 1.2 支持的版本

支持21.0.4版本

版本号查询方法，在Atlas产品环境下，运行命令：

```bash
npu-smi info
```
可以查询支持SDK的版本号


### 1.3 软件方案介绍

本方案中，采用yolov3预训练模型对输入图片进行车辆识别，车辆识别后对识别出的车辆图像进行抠图，然后使用GoogLeNet_cars模型进行车型识别，最终根据GoogLeNet_cars模型识别得到的车型信息和置信度生成框出并标有车辆车型与置信度的jpg图片。


### 1.4 代码目录结构与说明

本工程名称为VehicleIdentification，工程目录如下图所示：
```
├── models
│   ├── googlenet
│   │   ├── car.names
│   │   ├── googlenet.om
│   │   ├── updatemodel.py				# caffemodel旧版本升级新版本
│   │   ├── insert_op.cfg				# googlenet aipp转换配置
│   │   └── vehiclepostprocess.cfg		# googlenet后处理配置
│   ├── yolo
│   │   ├── aipp_yolov3_416_416.aippconfig
│   │   ├── coco.names
│   │   ├── yolov3_tf_bs1_fp16.cfg		# yolov3后处理配置
│   │   └── yolov3_tf_bs1_fp16.om
├── pipeline
│   └── identification.pipeline         # pipeline文件
├── vehiclePostProcess			        # 车型识别后处理库
│   ├── CMakeLists.txt
│   ├── VehiclePostProcess.cpp
│   └── VehiclePostProcess.h
├── input
│   ├── xxx.jpg							# 待检测文件
│   └── yyy.jpg
├── result
│   ├── xxx_result.jpg					# 运行程序后生成的结果图片
│   └── yyy_result.jpg
├── main.py
└── build.sh 							# 编译车型识别后处理插件脚本
```

### 1.5 技术实现流程图

![process](./img/process.png)

图1 车型识别流程图



![pipeline](./img/pipeline.png)

图2 车型识别pipeline示意图


### 1.6 适用场景
项目适用于光照条件较好，车辆重叠程度低，车辆轮廓明显，且图片较清晰的测试图片

**注**：由于GoogLeNet_cars模型限制，仅支持识别在`./models/vehicle/car.names`文件中的 **431** 种车辆。且由于此模型为2015年训练，在识别2015年之后外观有较大变化的车辆时误差较大。

## 2 环境依赖

| 软件名称 | 版本   |
| :--------: | :------: |
|Ubuntu|18.04.1 LTS   |
|Python|3.9.2|
|numpy|1.22.3|
|opencv-python|4.5.5|

mxVision 5.0.RC1
Ascend-CANN-toolkit （310使用6.3.RC1，310B使用6.2.RC1）

在编译运行项目前，需要设置环境变量：

- 环境变量介绍

```shell
. /usr/local/Ascend/ascend-toolkit/set_env.sh # Ascend-cann-toolkit开发套件包默认安装路径，根据实际安装路径修改
. ${MX_SDK_HOME}/mxVision/set_env.sh # ${MX_SDK_HOME}替换为用户的SDK安装路径
```


## 3 模型获取

### 3.1 yolo模型转换

**步骤1** 在ModelZoo上下载YOLOv3模型。[下载地址](https://www.hiascend.com/zh/software/modelzoo/detail/1/ba2a4c054a094ef595da288ecbc7d7b4)

**步骤2** 将获取到的YOLOv3模型pb文件存放至`./models/yolo/`。

**步骤3** 模型转换

在`./models/yolo`目录下执行一下命令

```bash
# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 执行，转换YOLOv3模型
# Execute, transform YOLOv3 model.

atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"
# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
```

执行完模型转换脚本后，会生成相应的.om模型文件。 执行后终端输出为：
```bash
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```


### 3.2 googlenet模型转换

**步骤1** 下载googlenet模型权重文件。[下载地址](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/googlenet_finetune_web_car_iter_10000.caffemodel)

**步骤2** 更新caffemodel文件：

由于此模型为老版本模型，atc不支持转换，需要将模型权重文件与结构文件更新，项目在[模型下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/VehicleIdentification/models.zip)中同时提供更新后的模型结构文件deploy.prototxt。

将权重文件放置于`./models/googlenet/`目录下，执行目录下的updatemodel.py（需要安装caffe环境），得到新版caffe权重文件`after-modify.caffemodel`。

> 若环境没有安装caffe，也可以在 [**此处**](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/VehicleIdentification/models.zip)  获得已经转换好的`after-modify.caffemodel`文件。

**步骤3** 模型转换：

在`./models/googlenet`目录下执行一下命令

```bash
# 环境变量配置如3.1

atc --framework=0 --model=./deploy.prototxt --weight=./after-modify.caffemodel --input_shape="data:1,3,224,224" --input_format=NCHW --insert_op_conf=./insert_op.cfg --output=./googlenet --output_type=FP32 --soc_version=Ascend310
```

执行完模型转换脚本后，会生成相应的.om模型文件。 执行后终端输出为：
```bash
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```



模型转换使用了ATC工具，如需更多信息请参考:

 https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md

备注：若推理芯片为310B，需要将atc-env脚本中模型转换atc命令中的soc_version参数设置为Ascend310B1。

### 3.3 可用模型获取

此处提供转换好的YOLOV3模型，车型识别模型（googlenet）的om文件：[**下载地址**](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/VehicleIdentification/models.zip)

注：**下载后请将两个模型请放置于models的对应目录下（`models/yolo`和`models/googlenet`）**



## 4 编译与运行

示例步骤如下：

**步骤1** 

后处理插件库编译：在项目目录下执行
```bash
bash build.sh
```

编译成功后，生成的`libvehiclepostprocess.so`后处理库文件位于`./lib`目录下。执行后终端输出为：

```bash
[ 50%] Building CXX object CMakeFiles/vehiclepostprocess.dir/VehiclePostProcess.cpp.o
[100%] Linking CXX shared library {项目路径}/lib/libvehiclepostprocess.so
[100%] Built target vehiclepostprocess
```

**步骤2**

修改libvehiclepostprocess.so文件权限为640：


**步骤3** 

自行选择一张或多张jpg文件，放入新建`./input`目录下，再执行
```bash
python3 main.py
```

执行后会在终端按顺序输出车辆的车型信息和置信度

生成的结果图片中添加方框框出车辆，在方框左上角标出车型信息和置信度，按 **{原名}_result.jpg** 的命名规则存储在`./result`目录下，查看结果文件验证检测结果。
