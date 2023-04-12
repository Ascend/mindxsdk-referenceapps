## 基于MindX SDK的faceswap应用开发

## 1 介绍
faceswap应用基于MindX SDK开发，在昇腾芯片上进行目标检测，脸部关键点推理以及目标替换，将替换结果可视化并保存。  

样例输入：两张脸部特征清晰，轮廓完整的正面人像jpg，face1.jpg,face2.jpg，且每张人像图片中仅包含一张目标，我们定义face1.jpg为源脸（底图），face2.jpg为目标脸（覆盖图）。<br/>
样例输出：一张脸部特征更换后的人像jpg，得到结果为face1图片中目标显示face2的目标特征。<br/>  
```
由于模型精度限制，关于输入输出的其他说明见本文 8. 常见问题
```
### 1.1 支持的产品

本项目以昇腾Atlas 500 A2为主要的硬件平台。

### 1.2 支持的版本

本样例配套的CANN版本为[6.2.RC1](https://www.hiascend.com/software/cann/commercial)  
支持的SDK版本为[5.0.RC1](https://www.hiascend.com/software/Mindx-sdk)  
MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)
### 1.3  技术实现流程图<br/>
本项目的主要工作流程为，初始输入目标图像和原图像，第一阶段采用yolov4模型进行目标检测，接着利用后处理插件、抠图插件对检测结果进行处理，得到目标区域部分；
第二阶段采用脸部特征点检测模型对得到的目标区域进行检测，得到脸部106个特征点的坐标，并在后处理阶段进行脸部替换，最后目标替换可视化结果。  
技术实现流程如下图所示：  

![alt 技术流程](Img4md/img.png)

## 2  代码目录结构<br/>
本Sample工程名称为faceswap,工程目录如下图所示：
```angular2html
|-------- data                                 // 输入存放文件夹（需用户新建，见5.4）
|-------- models
|           |---- yolov4.cfg                   // yolov4后处理配置文件（用于目标检测）
|           |---- coco.names                   // yolov4模型所有可识别类
|           |---- yolov4_detection.om          // yolov4离线推理模型，见4.1
|           |---- V3ONNX.cfg                   // 脸部特征点检测模型转换配置文件（用于检测脸部特征点）
|           |---- V3ONNXX.om                   // 脸部特征点检测离线模型，见4.2
|-------- pipline
|           |---- faceswap.pipeline            // 目标替换流水线配置文件
|-------- result                               // 存放结果文件（需用户新建，见5.5）
|-------- faceswap_main.py                              
|-------- faceswap_post.py                     // 后处理模块
|-------- README.md   
|-------- run.sh                               // 样例运行脚本
```  
## 3 环境依赖

| 软件名称 | 版本   | 
| :--------: | :------: |
|操作系统|18.04.1   |
|MindX SDK|5.0.RC1|
|Python| 3.9.2|
|numpy | 1.21.0 |
|opencv-python|4.5.2|

请注意MindX SDK使用python版本为3.9.2，如出现无法找到python对应lib库请在root下安装python3.9开发库  
`apt-get install libpython3.9`

## 4 模型获取及转换
### 4.1 目标检测模型转换
目标检测采用提供的离线模型yolov4_detection.om进行推理。由于文件上传限制，提供下载的[链接](https://mindx.sdk.obs.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/faceswap/yolov4_detection.om) 点击`链接`，下载yolov4_detection.om,并将模型存放在工程/model目录下。
### 4.2 脸部特征点检测模型
**步骤1**  
在[Github_pfld_106_face_landmarks](https://github.com/Hsintao/pfld_106_face_landmarks) 上选择v3.onnx，下载特征点检测对应的onnx模型：[v3.onnx](https://github.com/Hsintao/pfld_106_face_landmarks/blob/master/output/v3.onnx) [备份链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/faceswap/v3.onnx)；  
**步骤2**
将获取到的v3.onnx模型存放至”项目所在目录/model“文件夹下；   
**步骤3**
在model目录下执行以下命令：  
```
# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

. /usr/local/Ascend/ascend-toolkit/set_env.sh   # Ascend-cann-toolkit开发套件包默认安装路径，根据实际安装路径修改

# 执行如下命令，转换v3.onnx模型。
# Execute, transform v3.onnx model.
atc --model=v3.onnx --framework=5 --output=V3ONNXX --soc_version=Ascend310B1 --insert_op_conf=V3ONNX.cfg --out_nodes="Gemm_169:0"
```

其中--insert_op_conf参数为aipp预处理算子配置文件路径。该配置文件V3ONNX.cfg在输入图像进入模型前对其进行预处理。该配置文件保存在项目/model目录下。  
执行完模型转换脚本后，若提示如下信息说明模型转换成功，会在output参数指定的路径下生成V3ONNXX.om模型文件。  
```
ATC run success, welcome to the next use. 
```
注：模型转换使用了ATC工具，如需更多信息请参考[ATC](https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md)

## 5 测试  
1. 获取om模型   
```
见4： 模型转换
```
2. 配置
```   
#执行如下两个环境配置脚本设置环境变量，运行命令：
. /usr/local/Ascend/ascend-toolkit/set_env.sh   # Ascend-cann-toolkit开发套件包默认安装路径，根据实际安装路径修改
. ${MX_SDK_HOME}/mxVision/set_env.sh   # ${MX_SDK_HOME}替换为用户的SDK安装路径

```
3. 配置pipeline  
根据所需场景，配置pipeline文件，调整路径参数等。
>yolov4_detection模型推理及后处理配置
```
  # 配置mxpi_objectpostprocss0插件的配置文件路径（postProcessConfigPath）、标签路径（labelPath）、后处理路径（postProcessLibPath）
  "mxpi_objectpostprocessor0": {
           "props": {
                    "dataSource": "mxpi_tensorinfer0",
                    "postProcessConfigPath": "model/yolov4.cfg",
                    "labelPath": "model/coco.names",
                    "postProcessLibPath": "${SDK安装路径}/mxVision/lib/modelpostprocessors/libyolov3postprocess.so"
                },
              "factory": "mxpi_objectpostprocessor",
              "next": "mxpi_imagecrop0"
        },
```
4. 获取测试需要的测试图片  
```
# 执行以下命令，进入工程并在工程目录下新建data文件夹，用于存放测试图片
cd $home
cd faceswap
mkdir data
# 准备要替换的两张正面人像jpg图片，存放在工程/data目录下，并分别命名为face1.jpg, face2.jpg。
```
5. 在工程目录下执行如下命令，新建result文件夹，用于存放推理结果
```
cd home/faceswap
mkdir result
```
6. 运行可执行文件
```
切换至工程主目录，执行以下命令运行样例。命令行格式为 [python3 faceswap_main.py 源目标图片路径 目标目标图片路径]  

例：python3 faceswap_main.py data/face1.jpg data/face2.jpg
```

7. 查看结果  
执行`faceswap_main.py`完毕后，可在工程目录`result`中查看目标替换结果`face_swap_result.jpg`。  
8. 常见问题  
① 在目标检测阶段，由于yolov4_detection.om模型的coco.names标签集中同时存在people，face两类标签。当对输入图片的检测结果为people时，无法进行后续的换脸操作，故输入图片应尽可能体现脸部特征，建议输入图片为类似于证件照的半身人像。否则，当输入为全身人像时候，图片标签为people，无法进行后续换脸操作；  
② 在特征点检测阶段，由于特征点检测模型限制，输入目标应尽可能采用脸部清晰，轮廓完整的正面图像，即应尽可能输入2D目标图像，且脸部应尽可能避免眼镜等一类装饰物。若图片中存在3D特征，如输入侧脸时，可能会由于脸部特征点检测存在偏差导致换脸效果不佳；  
③ 针对MindX SDK固有插件的输入限制，输入目标图片的宽高均应限制在[32, 8192]区间内，否则会导致图片入流失败。当输入图片尺寸不符合要求时，系统会提示相应的错误信息。




