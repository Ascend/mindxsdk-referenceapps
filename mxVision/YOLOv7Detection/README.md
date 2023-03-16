# YOLOv7 模型推理参考样例
## 1. 介绍

YOLOv7 目标检测后处理插件基于 MindX SDK 开发，对图片中的不同类目标进行检测。输入一幅图像，可以检测得到图像中大部分类别目标的位置。本方案基于 pytorch 版本原始 yolo7x.pt 模型所转换的 om 模型进行目标检测，默认模型包含 80 个目标类。

### 1.1 支持的产品

本项目以昇腾310及310P芯片卡为主要的硬件平台。


### 1.2 支持的版本

支持的SDK版本为 5.0.RC1, CANN 版本为 6.0.RC1。


### 1.3 软件方案介绍 

封装yolov7后处理方法到后处理插件中，通过编译yolov7postprocessor插件so, 将该插件应用到pipeline或者v2接口进行后处理计算。

#### 1.3.1 业务流程加图像预处理方案

Pytorch框架对yolov7模型推理时，前处理方案包括解码为BGR->等比缩放->中心补边->转换为RGB->标准化，main.cpp中通过在310P场景下通过dvpp及opencv对应方法进行了相应的处理，标准化的步骤在aipp配置项中完成。                           

### 1.4 代码目录结构与说明

本工程名称为 YOLOv7Detection，工程目录如下所示：
```
.
├── run.sh                          # 编译运行main.cpp脚本
├── main.cpp                        # mxBasev2接口推理样例流程
├── plugin
│     ├── Yolov7PostProcess.h       # yolov7后处理插件编译头文件(需要被main.cpp引入)
│     ├── Yolov7PostProcess.cpp     # yolov7后处理插件实现
│     └── CMakeLists.txt            # 用于编译后处理插件
├── model
│     ├── coco.names                # 需要下载，下载链接在下方
│     └── yolov7.cfg                # 模型后处理配置文件，配置说明参考《mxVision用户指南》中已有模型支持->模型后处理配置参数->YOLOv5模型后处理配置参数
├── pipeline
│     ├── Sample.pipeline           # 参考pipeline文件，用户需要根据自己需求和模型输入类型进行修改
│     └── SampleYuv.pipeline        # 参考pipeline文件，用于配置yuv模型，用户需要根据自己需求和模型输入类型进行修改，
├── CMakeLists.txt                  # 编译main.cpp所需的CMakeLists.txt, 编译插件所需的CMakeLists.txt请查阅用户指南
├── test.jpg                        # 需要用户自行添加测试数据
└── README.md

```

注：coco.names文件源于[链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/contrib/Collision/model/coco.names)的coco2014.names文件，下载之后，放到models目录下。   
另外，yolov7.cfg中新增了一个配置项PADDING_TYPE用于区分补边的情况，若采用dvpp补边则填写0，采用opencv补边则填写1，默认为1。    
SampleYuv.pipeline中resize插件需要选用双线性插值的方式，需要根据310和310P环境填写interpolation的参数。


## 2 环境依赖

推荐系统为ubuntu 18.04，芯片环境310P：

在编译运行项目前，需要设置环境变量：

MindSDK 环境变量:

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
ascend-toolkit-path: CANN 安装路径。
```   

## 3. 模型转换    

关键依赖版本说明    
PyTorch >=1.8.0    

请参考[链接](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov7_for_Pytorch)对模型进行下载和转换为om。   
注意：由于main.cpp样例在310P环境下解码后的图片为BGR格式，因此使用aipp转换至om时，请将上述链接中的教程中 4. 使用aipp预处理 aipp_op中的rbuv_swap_switch项设置为true。
转换完成后，将该模型放到model路径下。  

转换为BGR输入参考
```
aipp_op{
   aipp_mode : static
   input_format : RGB888_U8
   src_image_size_w : 640 
   src_image_size_h : 640
   
   csc_switch : false
   rbuv_swap_switch : true

   min_chn_0 : 0
   min_chn_1 : 0
   min_chn_2 : 0
   var_reci_chn_0: 0.0039215686274509803921568627451
   var_reci_chn_1: 0.0039215686274509803921568627451
   var_reci_chn_2: 0.0039215686274509803921568627451

}
```    
转换为YUVSP420输入模型参考
```
aipp_op {
    aipp_mode : static
    input_format : YUV420SP_U8
    csc_switch : true
    rbuv_swap_switch : false
    matrix_r0c0 : 256
    matrix_r0c1 : 0
    matrix_r0c2 : 359
    matrix_r1c0 : 256
    matrix_r1c1 : -88
    matrix_r1c2 : -183
    matrix_r2c0 : 256
    matrix_r2c1 : 454
    matrix_r2c2 : 0
    input_bias_0 : 0
    input_bias_1 : 128
    input_bias_2 : 128

    min_chn_0 : 0
    min_chn_1 : 0
    min_chn_2 : 0
    var_reci_chn_0: 0.0039215686274509803921568627451
    var_reci_chn_1: 0.0039215686274509803921568627451
    var_reci_chn_2: 0.0039215686274509803921568627451
}
```

## 4. 编译与运行

### 4.1 mxBasev2接口推理业务流程

**步骤1**   
编译后处理插件so：  
后处理插件编译步骤参考《mxVision用户指南》中 深入开发->推理模型后处理开发介绍->新框架模型后处理->编译，其中"samplepostprocess"和"SamplePostProcess.cpp"分别代表生成的后处理动态库名和生成后处理的目标文件，对应到yolov7则yolov7postprocess和Yolov7PostProcess.cpp，

注意：  
修改CMakeLists.txt中 ```set(PLUGIN_NAME "samplepostprocess")``` 一行中插件名称，为 ```set(PLUGIN_NAME "yolov7postprocess")```    
修改CMakeLists.txt中 ```add_library(${TARGET_LIBRARY} SHARED SamplePostProcess.cpp)``` 一行中cpp文件名称，为 ```add_library(${TARGET_LIBRARY} SHARED Yolov7PostProcess.cpp)```   
生成的so会在make install时被安装到${MX_SDK_HOME}/lib/modelpostprocessors/下，请确保该so文件权限为440。   

**步骤2**    
放入待测图片。将一张图片放项目根路径下，命名为 test.jpg。   

**步骤3**   
对样例main.cpp中加载的模型路径、模型配置文件路径进行检查，确保对应位置存在相关文件，请参考run.sh中的说明。

**步骤4**    
图片检测。在项目路径根目录下运行命令：  

```
bash run.sh -m model_path -c model_config_path -l model_label_path -i image_path [-y]
```     
### 4.2 pipeline推理业务流程

请参考《mxVision用户指南》中 使用命令行开发->样例介绍->C++运行步骤 章节，使用senddata和getresult方式进行推理，请配置样例中pipeline路径为当前项目下pipeline/Sample.pipeline文件，并对该pipeline文件中的模型及其配置文件路径进行合理配置。
