# C++ 基于MxBase 的yolov5小麦检测

## 1 介绍
本开发样例是基于mxBase开发的端到端推理的小麦检测程序，实现对图像中的小麦进行识别检测的功能，并把可视化结果保存到本地。其中包含yolov5的后处理模块开发。
该Sample的主要处理流程为：
Init > ReadImage >Resize > Inference >PostProcess >DeInit

### 1.1 支持的产品

支持昇腾310芯片

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

推荐系统为ubantu 18.04，环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| :--------: | :------: |
|Ubantu|18.04|
|MindX SDK|2.0.2|


## 编译与运行

示例步骤如下：

**步骤1** 

修改CMakeLists.txt文件 将set(MX_SDK_HOME ${SDK安装路径}) 中的${SDK安装路径}替换为实际的SDK安装路径

**步骤2** 

ASCEND_HOME Ascend安装的路径，一般为/usr/local/Ascend
LD_LIBRARY_PATH 指定程序运行时依赖的动态库查找路径，包括ACL，开源软件库，libmxbase.so以及libyolov3postprocess.so的路径
```
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
```

**步骤3** 

cd到mxbase目录下，执行如下编译命令：
bash build.sh

**步骤4** 

制定jpg图片进行推理，将需要进行推理的图片放入mxbase 目录下的新文件夹中，例如mxbase/test。 eg:推理图片为xxx.jpg
cd 到mxbase 目录下
```
./mxBase_sample ./test/
```