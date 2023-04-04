# C++ 基于MxBase 的yolov5小麦检测

## 1 介绍
本开发样例是基于MindX SDK基础模块（MxBase）开发的端到端推理的小麦检测程序，实现对图像中的小麦进行识别检测的功能，并把可视化结果保存到本地。其中包含yolov5的后处理模块开发。
该Sample的主要处理流程为：
Init > ReadImage >Resize > Inference >PostProcess >DeInit

参考设计来源：https://github.com/Yunyung/Global-Wheat-Detection-Competition
数据集来源：https://www.kaggle.com/c/global-wheat-detection/data

### 1.1 支持的产品

支持昇腾310B芯片

### 1.2 软件方案介绍

请先总体介绍项目的方案架构。如果项目设计方案中涉及子系统，请详细描述各子系统功能。如果设计了不同的功能模块，则请详细描述各模块功能。

表1.1 系统方案中各模块功能：

| 序号 | 子系统            | 功能描述                                                     |
| ---- | ----------------- | ------------------------------------------------------------ |
| 1    | 资源初始化        | 调用mxBase::DeviceManager接口完成推理卡设备的初始化。        |
| 2    | 图像输入          | C++文件IO读取图像文件                                        |
| 3    | 图像解码/图像缩放 | 调用mxBase::DvppWrappe.DvppJpegDecode()函数完成图像解码，VpcResize()完成缩放。 |
| 4    | 模型推理          | 调用mxBase:: ModelInferenceProcessor 接口完成模型推理        |
| 5    | 后处理            | 获取模型推理输出张量BaseTensor，进行后处理。                 |
| 6    | 保存结果          | 输出图像当中小麦的bbox坐标以及置信度，保存标记出小麦的结果图像。           |
| 7    | 资源释放      | 调用mxBase::DeviceManager接口完成推理卡设备的去初始化。      |



### 1.3 代码目录结构与说明

本sample工程名称为**mxBase_wheatDetection**，工程目录如下图所示：
```
|-------- model
|           |---- onnx_best_v3.om       // 小麦检测om模型
|           |---- aipp.aippconfig       // 模型转换aipp配置文件
|           |---- coco.names       		// 标签文件
|-------- yolov5Detection				// 小麦检测模型推理文件
|           |---- Yolov5Detection.cpp       
|           |---- Yolov5Detection.h         
|-------- yolov5PostProcess  			// 小麦检测后处理文件
|           |---- Yolov5Detection.cpp       
|           |---- Yolov5Detection.cpp       
|-------- build.sh                            // 编译文件
|-------- main.cpp                            // 主程序  
|-------- CMakeLists.txt                      // 编译配置文件   
|-------- README.md   
```


### 1.4 技术实现流程图

![image-2021092301](image-2021092301.jpg)


## 2 环境依赖

推荐系统为ubantu 18.04。

| 软件名称 | 版本   |
| -------- | ------ |
| python    | 3.9.2     | 
| MindX SDK     |    5.0RC1    |
| CANN | 310使用6.3.RC1<br>310B使用6.2.RC1 |

## 模型转换

**步骤1** 模型获取

在Kaggle上下载YOLOv5模型 。[下载地址](https://www.kaggle.com/yunyung/yolov5-wheat)

在Github上下载YOLOv5的各个文件。[下载地址](https://github.com/ultralytics/yolov5)

这里提供了已经转好的416*416尺寸的onnx和om模型，以及上述两个下载链接的模型和文件。[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/mxBase_wheatDetection/model.zip)

将下载的YOLOv5模型pt文件通过YOLOv5自带的export模型转换函数转换为onnx格式的文件（onnx模型已提供）
```
python export.py --weights best_v3.pt --img 416 --batch 1 --simplify
```

**步骤2** 模型存放

将获取到的YOLOv5模型onnx文件放至上一级的model文件夹中

**步骤3** 执行模型转换命令

(1) 配置环境变量
#### 设置环境变量
#### Set environment PATH (Please confirm that the install_path is correct).

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh #toolkit默认安装路径，根据实际安装路径修改
. ${SDK_INSTALL_PATH}/mxVision/set_env.sh
```

(2) 转换模型

```
atc --model=./best_v3_t.onnx --framework=5 --output=./onnx_best_v3 --soc_version=Ascend310B1 --insert_op_conf=./aipp.aippconfig --input_shape="images:1,3,416,416" --output_type="Conv_1228:0:FP32;Conv_1276:0:FP32;Conv_1324:0:FP32" --out_nodes="Conv_1228:0;Conv_1276:0;Conv_1324:0"
```
## 使用场景概括

### 适用条件

适用于各个生长阶段的小麦，必须是以麦穗的形式，可在任何天气条件下进行识别，视角包含正视、俯视、仰视。

### 限制条件

尺寸条件：单个小麦的宽高像素不得超过500

光照条件：光线较为良好，如果光线不足，必须有小麦的完整轮廓

小麦条件：小麦应为根、茎、叶、穗的形式，不能为小麦粒

## 参数调节

| 参数名称 |参数介绍| 修改方法   | 默认值   |
| -------------- | --------------------------------------------- | --------------------------------------------------------------------- | -------- |
|CONFIDENCE      |置信度|在mxBase_wheatDetection/yolov5Detection/yolov5Detection.cpp文件中，修改CONFIDENCE的大小即可| 0.5 |
|objectnessThresh|是否为目标的阈值，大于阈值即认为是目标 |在mxBase_wheatDetection/main.cpp文件中，修改initParam.objectnessThresh的大小即可|0.3|
|iouThresh       |两个框的IOU阈值，超过阈值即认为同一个框,用于nms算法|在mxBase_wheatDetection/main.cpp文件中，修改initParam.iouThresh的大小即可|0.3|
|scoreThresh     |是否为框的阈值，大于阈值即认为是框|在mxBase_wheatDetection/main.cpp文件中，修改initParam.scoreThresh的大小即可|0.45|

## 编译与运行

示例步骤如下：

**步骤1** 

cd到mxBase_wheatDetection目录下，执行如下编译命令：

```
bash build.sh
```

**步骤2** 

制定jpg图片进行推理，将需要进行推理的图片放入mxBase_wheatDetection目录下的新文件夹中，例如mxBase_wheatDetection/test。 eg:推理图片为xxx.jpg
cd 到mxBase_wheatDetection目录下
```
./mxBase_wheatDetection ./test/
```

## 精度测试

使用(https://github.com/Cartucho/mAP)进行map计算，需要参考readme将推理结果数据以下方式转换并存储为txt

```
<CLASS_NAME> <LEFT> <TOP> <RIGHT> <BOTTOM>
```

并将相关txt保存至input/detection-results/文件夹，对标签进行相同转换并保存至 input/ground-truth/文件夹。
随后执行python main.py以计算map并生成相关绘图。
使用GWHD2020数据集训练集进行测试，map为0.8，性能为14FPS.