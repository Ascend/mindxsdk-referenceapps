# C++ 基于MxBase 的人群计数图像检测样例及后处理模块开发

## 1 介绍
本开发样例是基于mxBase开发的端到端推理的C++应用程序，可在昇腾芯片上进行人群计数目标检测，并把可视化结果保存到本地。开发端到端人群计数-人群密度估计的参考设计，实现图像当中人计数的功能，并达到精度和性能要求。 该Sample的主要处理流程为： Init > ReadImage >Resize > Inference >PostProcess >DeInit

### 1.1 支持的产品

支持昇腾310芯片

### 1.2 支持的版本

在Atlas产品环境下，通过运行命令：

```
npu-smi info
```

可以查询支持SDK的版本号

### 1.3 软件方案介绍

人群计数项目实现：输入类型是图片数据（例如jpg格式的图片），通过调用MindX SDK mxBase提供的接口，使用DVPP进行图像解码，解码后获取图像数据，然后经过图像尺寸大小变换，满足模型的输入尺寸要求；将尺寸变换后的图像数据输入人群计数模型进行推理，模型输出经过后处理后，得到人群密度估计图和人计数估计值，输出人计数的估计值。

整个流程需要参考Ascend的参考样例：crowd_count_picture 样例，详见以下链接：https://gitee.com/ascend/samples/tree/master/python/contrib/crowd_count_picture  crowd_count_picture 样例是基于ACL实现的，本任务需要参考crowd_count_picture 样例，基于MindX SDK mxBase的接口实现。MindX SDK mxBase是对ACL接口的封装，提供了极简易用的API， 使能AI应用的开发。

表1.1 系统方案中各模块功能：

| 序号 | 子系统            | 功能描述                                                     |
| ---- | ----------------- | ------------------------------------------------------------ |
| 1    | 设备初始化        | 调用mxBase::DeviceManager接口完成推理卡设备的初始化。        |
| 2    | 图像输入          | C++文件IO读取图像文件                                        |
| 3    | 图像解码/图像缩放 | 调用mxBase::DvppWrappe.DvppJpegDecode()函数完成图像解码，VpcResize()完成缩放。 |
| 4    | 模型推理          | 调用mxBase:: ModelInferenceProcessor 接口完成模型推理        |
| 5    | 后处理            | 获取模型推理输出张量BaseTensor，进行后处理。                 |
| 6    | 保存结果          | 输出图像当中的人的数量，保存标记出人数的结果图像。           |
| 7    | 设备去初始化      | 调用mxBase::DeviceManager接口完成推理卡设备的去初始化。      |

### 1.4 代码目录结构与说明

本sample工程名称为 **CrowdCounting**，工程目录如下图所示：

![image-20210813133211603](image-20210813133211603.png)



### 1.5 技术实现流程图

![image-20210813133142529](image-20210813133142529.png)





## 2 环境依赖

请列出环境依赖软件和版本。

eg：推荐系统为ubuntu 18.04或centos 7.6，环境依赖软件和版本如下表：

| 软件名称 | 版本         |
| -------- | ------------ |
| 系统软件 | ubuntu 18.04 |

在编译运行项目前，需要设置环境变量：

- 环境变量介绍

```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

## 3 模型转换

**步骤1** 

下载原始模型权重下载：

[https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/count_person.caffe.caffemodel](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC Model/crowdCount/count_person.caffe.caffemodel)

**步骤2** 

下载原始模型网络：

[https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/count_person.caffe.prototxt](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC Model/crowdCount/count_person.caffe.prototxt)

**步骤3**

下载对应的cfg文件：

[https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/insert_op.cfg](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC Model/crowdCount/insert_op.cfg)

**步骤4**

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --input_shape="blob1:1,3,800,1408" --weight="count_person.caffe.caffemodel" --input_format=NCHW --output="count_person.caffe" --soc_version=Ascend310 --insert_op_conf=insert_op.cfg --framework=0 --model="count_person.caffe.prototxt" 
```

## 4 编译与运行
**步骤1** 

修改CMakeLists.txt文件 将set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")中的"$ENV{MX_SDK_HOME}"替换为实际的SDK安装路径

**步骤2** 

设置环境变量 ASCEND_HOME Ascend安装的路径，一般为/usr/local/Ascend LD_LIBRARY_PATH 指定程序运行时依赖的动态库查找路径，包括ACL，开源软件库，libmxbase.so以及模型后处理开发的动态链接库路径，比如：libyolov3postprocess.so

```
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/
opensource/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
```

**步骤3** 

cd到CrowdCounting目录下，执行如下编译命令： bash build.sh

**步骤4**

下载人群计数的图像：

```
wget https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/crowdCount/crowd.jpg --no-check-certificate
```

准备一张推理图片放入CrowdCounting目录下，执行：

```
./crowd_counting  ./crowd.jpg
```

## 5 软件依赖说明

| 依赖软件 | 版本  | 说明                                                         |
| -------- | ----- | ------------------------------------------------------------ |
| mxVision | 2.0.2 | 提供昇腾计算语言(AscendCL)的高级编程API，简化插件和推理应用开发。 |



