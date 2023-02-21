# PPYOLOE+ 模型推理参考样例
## 1. 介绍

PPYOLOE+ 目标检测后处理插件基于 MindX SDK 开发，对图片中的不同类目标进行检测。输入一幅图像，可以检测得到图像中大部分类别目标的位置。本方案基于 paddlepaddle 版本原始 ppyoloe_plus_crn_l_80e_coco_w_nms 模型所剪枝并转换的 om 模型进行目标检测，默认模型包含 80 个目标类。

### 1.1 支持的产品

本项目以昇腾310及310P芯片卡为主要的硬件平台。


### 1.2 支持的版本

支持的SDK版本为 5.0.RC1, CANN 版本为 6.0.RC1。


### 1.3 软件方案介绍 

封装ppyoloe后处理方法到后处理插件中，通过编译ppyoloepostprocessor插件so, 将该插件应用到pipeline或者v2接口进行后处理计算。

#### 1.3.1 业务流程加图像预处理方案

paddlepaddle框架的ppyoloe模型推理时，前处理方案包括解码为BGR->拉伸缩放->转换为RGB，main.cpp中通过在310P场景下通过dvpp对应方法进行了相应的处理。                            

### 1.4 代码目录结构与说明

本工程名称为 PPYOLOEPlusDetection，工程目录如下所示：
```
.
├── run.sh                          # 编译运行main.cpp脚本
├── main.cpp                        # mxBasev2接口推理样例流程
├── PPYoloePostProcess.h            # ppyoloe后处理插件编译头文件(需要被main.cpp引入)
├── PPYoloePostProcess.cpp          # ppyoloe后处理插件实现
├── model
│     ├── coco.names                # 需要下载，下载链接在下方
│     └── ppyoloe.cfg               # 模型后处理配置文件，配置说明参考《mxVision用户指南》中已有模型支持->模型后处理配置参数->YOLOv5模型后处理配置参数
├── pipeline
│     └── Sample.pipeline           # 参考pipeline文件，用户需要根据自己需求和模型输入类型进行修改
├── test.jpg                        # 需要用户自行添加测试数据
├── CMakeLists.txt                  # 编译main.cpp所需的CMakeLists.txt, 编译插件所需的CMakeLists.txt请查阅用户指南  
└── README.md

```

注：coco.names文件源于[链接](../Collision/model/coco.names)的coco2014.names文件，下载之后，放到models目录下。



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

**步骤1** 
建议通过[链接](https://github.com/PaddlePaddle/PaddleYOLO/blob/develop/docs/MODEL_ZOO_cn.md#PP-YOLOE)中 部署模型->PP-YOLOE+_l ->导出后的权重->(w/nms)下载paddle模型。
**步骤2** 
参考工具[链接](https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/tools/paddle)对下载后的paddle模型进行剪枝。对于PP-YOLOE+_l(w/nms)模型而言，建议将输出端口修改为"tmp20"和"concat_14.tmp_0"。
参考命令：
```
python prune_paddle_model.py --model_dir ${input_model_dir} --model_filename ${pdmodel_file_name} --params_filename ${pdiparams_file_name} --output_names tmp_20 concat_14.tmp_0 --save_dir ${new_model_dir}
```    
其中：  
```${input_model_dir}``` 代表输入模型根目录，例如 ```./ppuoloe_plus_crn_l_80e_coco_w_nms```   
```${pdmodel_file_name}``` 代表模型模型目录下模型名称，例如 ```model.pdmodel```   
```${pdiparams_file_name}``` 代表模型模型目录下模型参数，例如 ```model.pdiparams```   
```${new_model_dir}``` 代表模型输出的路径     

**步骤3**   

参考工具[链接](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README.md)转换为onnx模型

**步骤4** 
将onnx模型转换为om模型
配置aipp.cfg参考:
```
aipp_op {
    aipp_mode : static
    input_format : RGB888_U8
    src_image_size_w : 640
    src_image_size_h : 640

    csc_switch : false
    rbuv_swap_switch : true
}
```
atc转换模型命令
```
atc --framework=5 --model=${onnx_model} --output={output_name} --input_format=NCHW --input_shape="image:1, 3, 640, 640" --log=error --soc_version={soc_name} --insert_op_conf=${aipp_cfg_file} --output_type=FP32
```
其中：
```${onnx_model}``` 代表输入onnx模型，例如 ```model.onnx```    
```${output_name}``` 代表输出模型名称，例如 ```ppyoloe```    
```${soc_name}``` 代表芯片型号，例如 ```Ascend310P3```    
```${aipp_cfg_file}``` 代表模型输出的路径, 例如 ```aipp.cfg```     

**步骤4** 

转换完成后，将该om模型放到model路径下。

## 4. 编译与运行

### 4.1 mxBasev2接口推理业务流程

**步骤1** 编译后处理插件so：  

后处理插件编译步骤参考《mxVision用户指南》中 深入开发->推理模型后处理开发介绍->新框架模型后处理->编译，其中"samplepostprocess"和"SamplePostProcess.cpp"分别代表生成的后处理动态库名和生成后处理的目标文件，对应到ppyoloe则为ppyoloepostprocess和PPYoloePostProcess.cpp，

注意：  
修改CMakeLists.txt中 ```set(PLUGIN_NAME "samplepostprocess")``` 一行中插件名称，为 ```set(PLUGIN_NAME "ppyoloepostprocess")```
修改CMakeLists.txt中 ```add_library(${TARGET_LIBRARY} SHARED SamplePostProcess.cpp)``` 一行中cpp文件名称，为 ```add_library(${TARGET_LIBRARY} SHARED PPYoloePostProcess.cpp)```
生成的so会在make install时被安装到${MX_SDK_HOME}/lib/modelpostprocessors/下，请确保该so文件权限为440。

**步骤2**  
放入待测图片。将一张图片放项目根路径下，命名为 test.jpg。

**步骤3**   
对main.cpp样例中加载的模型路径、模型配置文件路径进行检查，确保对应位置存在相关文件，包括：  
string modelPath = "models/ppyoloe.om";  
string ppyoloeConfigPath = "models/ppyoloe.cfg";  
string ppyoloeLabelPath = "models/coco.names";  

**步骤4**   
图片检测。在项目路径根目录下运行命令：

```
bash run.sh
```     
### 4.2 pipeline推理业务流程

请参考《mxVision用户指南》中 使用命令行开发->样例介绍->C++运行步骤 章节，使用senddata和getresult方式进行推理，请配置样例中pipeline路径为当前项目下pipeline/Sample.pipeline文件，并对该pipeline文件中的模型及其配置文件路径进行合理配置。
