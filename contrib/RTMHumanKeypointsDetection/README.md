# 实时人体关键点检测

## 1 介绍

人体关键点检测是指在输入图像上对指定的 18 类人体骨骼关键点位置进行检测，然后将关键点正确配对组成相应的人体骨架，展示人体姿态。本项目基于MindX SDK，在昇腾平台上，实现了对RTSP视频流进行人体关键点检测并连接成人体骨架，最后将检测结果可视化并保存。

### 1.1 支持的产品

昇腾310(推理)

### 1.2 支持的版本
支持的SDK版本为 5.0.RC1, CANN 版本310使用6.3.RC1，310B使用6.2.RC1。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 软件方案介绍

本系统设计了不同的功能模块。主要流程为：视频拉流传入业务流中，然后通过解码插件对视频进行解码，再对解码出来的YUV图像进行尺寸调整，然后利用OpenPose模型进行人体关键点检测，然后我们自己编写的后处理插件会把人体关键点信息传递给绘图插件，绘图完毕后进行视频编码，最后把结果输出。各模块功能描述如表2.1所示：

表2.1 系统方案中各模块功能：

| 序号 | 子系统     | 功能描述                                                     |
| :--- | :--------- | :----------------------------------------------------------- |
| 1    | 视频拉流   | 调用MindX SDK的 **mxpi_rtspsrc**接收外部调用接口的输入视频路径，对视频进行拉流 |
| 2    | 视频解码   | 调用MindX SDK的**mxpi_videodecoder**                         |
| 3    | 图像缩放   | 调用**mxpi_imageresize**对解码后的YUV格式的图像进行指定宽高的缩放 |
| 4    | 检测推理   | 使用已经训练好的OpenPose模型，检测出图像中的车辆信息。插件：**mxpi_tensorinfer** |
| 5    | 模型后处理 | 使用自己编译的**mxpi_rtmopenposepostprocess**插件的后处理库libmxpi_rtmopenposepostprocess.so，进行人体关键点检测的后处理 |
| 6    | 绘图       | 调用OSD基础功能在YUV图片上绘制直线。插件：**mxpi_opencvosd** |
| 7    | 视频编码   | 调用MindX SDK的**mxpi_videoencoder**进行视频编码             |
| 8    | 输出       | 调用MindX SDK的**appsink**进行业务流结果的输出               |

### 1.4 代码目录结构与说明

本工程名称为RTMHumanKeypointsDetection，工程目录如下图所示：

```
├── eval
│   ├── pipeline
│   ├── plugin
│   ├── proto
│   └── eval.py
├── image
│   ├── pipeline.png
├── models
│   └── insert_op.cfg
├── pipeline
│   └── rtmOpenpose.pipeline         	# pipeline文件
├── plugins			        			# 实时人体关键点检测后处理库
│   ├── build.sh
│   ├── CMakeLists.txt
│   ├── MxpiRTMOpenposePostProcess.cpp
│   └── MxpiRTMOpenposePostProcess.h
├── test
│   └── main.cpp
├── CMakeLists.txt
├── README.md
├── build.sh 							# 编译
├── main.cpp
└── run.sh								# 运行
```

### 1.5 技术实现流程图

技术流程图如下：

![pipeline](image/pipeline.png)

### 1.6 特性及适用场景

使用测试视频应当人物清晰、光线充足、无环境背景干扰，而且人物在画面中占据范围不应太小、人物姿态不应过于扭曲、人物不应完全侧对镜头、背景不应太复杂；视频切勿有遮挡，不清晰等情况。

## 2 环境依赖

推荐系统为ubuntu 22.04，环境依赖软件和版本如下表：

| 依赖软件  | 版本               | 说明                          |
| --------- | ------------------ | ----------------------------- |
| ubuntu    | ubuntu 22.04.6 LTS | 操作系统                      |
| CANN      | 5.0.5.alpha001     | Ascend-cann-toolkit开发套件包 |
| MindX SDK | 3.0.RC2            | mxVision软件包                |

### 2.1 依赖安装

- CANN获取[链接](https://www.hiascend.com/software/cann)，安装[参考链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/505alpha001/softwareinstall/instg/atlasdeploy_03_0002.html)
- MindX SDK获取[链接](https://www.hiascend.com/software/Mindx-sdk)，安装[参考链接](https://www.hiascend.com/document/detail/zh/mind-sdk/30rc2/overview/index.html)

### 2.2 环境变量设置

确保环境中正确安装mxVision SDK。

在编译运行项目前，需要设置环境变量：

MindX SDK 环境变量:

```
. ${SDK-path}/set_env.sh
```

CANN 环境变量：

```
. ${ascend-toolkit-path}/set_env.sh
```

- 环境变量介绍

```
SDK-path: SDK mxVision 安装路径
ascend-toolkit-path: CANN 安装路径
```

### 2.3 第三方软件依赖

| 依赖软件 | 说明                           | 使用教程                                                     |
| -------- | ------------------------------ | ------------------------------------------------------------ |
| live555  | 实现视频转rstp进行推流         | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/Live555离线视频转RTSP说明文档.md) |
| ffmpeg   | 实现mp4格式视频转为264格式视频 | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/pc端ffmpeg安装教程.md#https://ffmpeg.org/download.html) |

## 3 模型转换

若使用A200I DK A2运行，推荐使用PC转换模型，具体方法可参考A200I DK A2资料。

本项目主要用到了用于人体关键点检测的OpenPose模型，OpenPose模型相关文件可以[点击此处](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)下载。本项目在MindX SDK中需要使用om模型，我们需要对模型进行转换。首先需要将pytorch模型转换为onnx模型，然后使用ATC模型转换工具将onnx模型转换为om模型。

本项目提供了输入尺寸为（512，512）的onnx模型，[点击此处下载](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/RTMHumanKeypointsDetection/human-pose-estimation512.onnx)，如不需要修改模型输入尺寸，可直接执行步骤二。

**步骤一** pth模型转onnx模型

1. 获取“lightweight-human-pose-estimation.pytorch”代码包文件
   方法一：通过git方式获取
   在服务器上，进入指定目录，执行以下命令：

   ```
   git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git
   ```

   将在当前目录下生成“lightweight-human-pose-estimation.pytorch”目录及目录中的相关文件；

   方法二：通过网页获取
   点击访问[pytorch 参考项目](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)，通过下载zip文件方式下载“lightweight-human-pose-estimation.pytorch”代码压缩包，上传至服务器指定位置后解压得到“lightweight-human-pose-estimation.pytorch-master”目录及目录中的相关文件；
   注：模型转换后续步骤采用“通过网页获取”得到的文件目录进行说明。

2. 安装相关依赖
   进入“lightweight-human-pose-estimation.pytorch-master”目录，执行以下命令安装相关依赖：

   ```
   pip install -r requirements.txt
   ```

3. 代码预处理
   进入“lightweight-human-pose-estimation.pytorch-master/scripts/”目录下，对“convert_to_onnx.py”文件做以下修改：

   ```python
   def convert_to_onnx(net, output_name):
       input = torch.randn(1, 3, 512, 512)   # 模型输入宽高，可自定义，此处修改为"512,512"
       input_names = ['data']
       output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                       'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']
       torch.onnx.export(net, input, output_name, verbose=True, opset_version=11, input_names=input_names, output_names=output_names)   # 使用pytorch将pth导出为onnx，此处增加ATC转换工具支持的ONNX算子库版本（如opset_version=11）
   ```

4. onnx模型转换

   通过[pth模型链接](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth)下载获取“checkpoint_iter_370000.pth”模型文件并拷贝至“lightweight-human-pose-estimation.pytorch-master”目录下，并在拷贝目标目录下执行以下命令将pth模型转换成onnx模型：

   ```
   python scripts/convert_to_onnx.py --checkpoint-path=checkpoint_iter_370000.pth
   ```

   注：
   	1）"--checkpoint-path"为下载的pth模型文件所在目录；若想指定输出的 onnx 模型文件名称，可增加"--output-name"参数，若不指定，则默认输出"human-pose-estimation.onnx"；
   	2）若服务器不存在"GPU"卡，可参考“6 常见问题”中的相关解决方法。

**步骤二** onnx模型转om模型

1.    进入"\${RTMHumanKeypointsDetection代码包目录}/models/"目录，对"insert_op.cfg"文件做以下修改：

   ```
   related_input_rank: 0
   src_image_size_w: 512 # onnx模型输入的宽，请根据对应模型进行修改
   src_image_size_h: 512 # onnx模型输入的高，请根据对应模型进行修改
   crop: false
   ```

2. 将“步骤一”转换得到的onnx模型拷贝至"${RTMHumanKeypointsDetection代码包目录}/models/"目录下，并在拷贝目标目录下执行以下命令将onnx模型转换成om模型：

   ```
   atc --model=./human-pose-estimation512.onnx --framework=5 --output=openpose_pytorch_512 --soc_version=Ascend310 --input_shape="data:1, 3, 512, 512" --input_format=NCHW --insert_op_conf=./insert_op.cfg
   ```

   注：
   	--model 为输入的onnx模型文件及所在目录
   	--output 为输出的om模型的名称
   	--input_shape 为指定的模型输入宽、高
   

等待该命令执行完毕，可能需要一些时间，执行完成后会在当前目录下生成项目需要的om模型文件。执行后终端输出为：

   ```
   ATC start working now, please wait for a moment.
   ATC run success, welcome to the next use.
   ```

表示转换成功。

   

## 4 编译与运行

**步骤1** 按照3.3小结**第三方软件依赖**安装live555和ffmpeg，按照 [Live555离线视频转RTSP说明文档](https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/reference_material/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md#https://gitee.com/link?target=https%3A%2F%2Fmindx.sdk.obs.cn-north-4.myhuaweicloud.com%2Ftool%2Flive555.tar.gz)将mp4视频转换为h264格式，并配置rtsp流地址，然后修改`RTMHumanKeypointsDetection/pipeline`目录下的rtmOpenpose.pipeline文件中mxpi_rtspsrc0的内容。

```
        "mxpi_rtspsrc0": {
            "factory": "mxpi_rtspsrc",
            "props": {
                "rtspUrl":"rtsp://xxx.xxx.xxx.xxx:xxxx/xxx.264",      // 修改为自己所使用的的服务器和文件名
                "channelId": "0"
            },
            "next": "mxpi_videodecoder0"
        },
```

**步骤2** 本项目需要使用 `mxpi_opencvosd` 插件，使用前需要生成所需的模型文件。执行MindX SDK开发套件包安装目录下 `operators/opencvosd/generate_osd_om.sh` 脚本生成所需模型文件。

**步骤3** 按照第3节**模型转换**中的步骤获得om模型文件，放在`models/`目录下。注意检查om模型文件名是否和`pipeline/rtmOpenpose.pipeline`中的mxpi_tensorinfer0 插件 modelPath 属性值相同，若不同需改为一致。

```
		"mxpi_tensorinfer0":{
            "next":"mxpi_rtmopenposepostprocess0",
            "factory":"mxpi_tensorinfer",
            "props":{
                "dataSource": "mxpi_imageresize0",
                "modelPath":"./models/openpose_pytorch_512.om"	//检查om模型文件名是否正确
            }
        },
```

若修改了模型的输入尺寸，还需要将 mxpi_imageresize0 插件中的 resizeWidth 和 resizeHeight 属性改成修改后的模型输入尺寸值；将 mxpi_rtmopenposepostprocess0 插件中的 inputWidth 和 inputHeight 属性改成修改后的模型输入尺寸值。

```
		"mxpi_imageresize0":{
            "next":"queue3",
            "factory":"mxpi_imageresize",
            "props":{
                "interpolation":"2",
                "resizeWidth":"512",	//输入的宽，请根据对应模型进行修改
                "resizeHeight":"512",	//输入的高，请根据对应模型进行修改
                "dataSource":"mxpi_videodecoder0",
                "resizeType":"Resizer_KeepAspectRatio_Fit"
            }
        },
```

```
		"mxpi_rtmopenposepostprocess0":{
            "next":"queue4",
            "factory":"mxpi_rtmopenposepostprocess",
            "props":{
                "imageSource":"mxpi_videodecoder0",
                "inputHeight":"512",	//输入的高，请根据对应模型进行修改
                "dataSource":"mxpi_tensorinfer0",
                "inputWidth":"512"		//输入的宽，请根据对应模型进行修改
            }
        },
```

**步骤4** 将pipeline里面的 mxpi_videoencoder0 插件中的 imageHeight 和 imageWidth 更改为上传视频的实际高和宽。

```
        "mxpi_videoencoder0":{
            "props": {
                "inputFormat": "YUV420SP_NV12",
                "outputFormat": "H264",
                "fps": "1",
                "iFrameInterval": "50",
                "imageHeight": "720",		#上传视频的实际高
                "imageWidth": "1280"		#上传视频的实际宽
            },
```

**步骤5** 编译。在`plugins/`目录里面执行命令：

```bash
bash build.sh
```

注：其中SDK安装路径`${SDK安装路径}`需要替换为用户的SDK安装路径

**步骤6** 运行。回到项目主目录下执行命令：

```bash
bash run.sh
```

命令执行成功后会在当前目录下生成结果视频文件`out.h264`，查看文件验证检测结果。

## 5 性能和精度测试

### 5.1 性能测试

使用`/test`目录下的main.cpp替换项目目录下的main.cpp，然后按照第4小结编译与运行中的步骤进行编译运行，服务器会输出运行到该帧的平均帧率，运行日志如下：

```
fps:63.9352
I20220729 16:59:22.297983 27442 MxpiVideoEncoder.cpp:324]Plugin(mxpi videoencodero)fps (65).
I20220729 16:59:22.302567 27448 MxpiRTMOpenposePostProcess.cpp:787]MxpiRTMOpenposePostProcess:Process start
I20220729 16:59:22.312167 27448 MxpiRTMOpenposePostProcess.cpp:834]MxpiRTMOpenposePostProcess:Process end
I2022072916:59:22.31323827427main.cpp:116]Dealing frame id:2104
fps:63.9351
I20220729 16:59:22.317531 27448 MxpiRTMOpenposePostProcess.cpp:787]MxpiRTMOpenposePostProcess:Process start
I20220729 16:59:22.327153 27448 MxpiRTMOpenposePostProcess.cpp:834]MxpiRTMOpenposePostProcess:Process end
I2022072916:59:22.32923327427main.cpp:116]Dealing frame id:2105
fps:63.9344
I20220729 16:59:22.332511 27448 MxpiRTMOpenposePostProcess.cpp:787]MxpiRTMOpenposePostProcess:Process start
I20220729 16:59:22.341825 27448 MxpiRTMOpenposePostProcess.cpp:834]MxpiRTMOpenposePostProcess:Process end
I2022072916:59:22.34488227427main.cpp:116]Dealing frame id:2106
fps:63.9344
I20220729 16:59:22.347272 27448 MxpiRTMOpenposePostProcess.cpp:787]MxpiRTMOpenposePostProcess:Process start
I20220729 16:59:22.356643 27448 MxpiRTMOpenposePostProcess.cpp:834]MxpiRTMOpenposePostProcess:Process end
I2022072916:59:22.36108327427main.cpp:116]Dealing frame id:2107
fps:63.9333
I20220729 16:59:22.362097 27448 MxpiRTMpenposePostProcess.cpp:787]MxpiRTMOpenposePostProcess:Process start
I20220729 16:59:22.371462 27448 MxpiRTMOpenposePostProcess.cpp:834]MxpiRTMOpenposePostProcess:Process end
I2022072916:59:22.37468327427main.cpp:116]Dealing frame id:2108
fps:63.9373
I20220729 16:59:22.376840 27448 MxpiRTMOpenposePostProcess.cpp:787]MxpiRTMOpenposePostProcess:Process start
I20220729 16:59:22.386106 27448 MxpiRTMOpenposePostProcess.cpp:834]MxpiRTMOpenposePostProcess:Process end
I2022072916:59:22.39022427427main.cpp:116]Dealing frame id:2109
```

通过日志结果，我们可以看出项目实际可以处理帧率在60fps左右的视频分析，而项目要求达到的性能是25fps，所以满足项目的性能要求。

注：输入视频帧率应高于25，否则无法发挥全部性能。

### 5.2 精度测试

使用COCO评测工具进行精度测试，数据集使用COCO VAL 2017数据集，使用输入为512*512的模型。具体步骤如下：

1. 准备数据集。[点击此处](http://images.cocodataset.org/zips/val2017.zip)下载测试集图片，[点击此处](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)下载测试集标注，在 `eval` 目录下创建 `dataset` 目录，解压到 `eval/dataset/` 目录下，结构如下：

   ```
   ├── eval
   │   ├── dataset
   │   │   ├── annotations
   │   │   │   └── person_keypoints_val2017.json
   │   │   └── val2017
   │   │       ├── 000000000139.jpg
   │   │       ├── 000000000285.jpg
   │   │       └── other-images
   
   ```

2. 安装测试工具。

   ```bash
   pip install pycocotools
   ```

3. 编译proto。在 `eval/proto/` 目录中执行命令：

   ```bash
   bash build.sh
   ```

4. 编译用于精度测试的后处理插件。在 `eval/plugins/` 目录中执行命令：

   ```bash
   bash build.sh
   ```
   
5. 运行精度测试脚本。在`eval/`目录中执行命令：

   ```bash
   python eval.py
   ```

   生成检测结果文件，并输出 COCO 格式的评测结果：
   
   ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.398
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.675
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.398
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.336
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.488
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.451
    Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.701
    Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.459
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.356
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.586
   ```
   
   可以看出，使用输入为512*512的模型。模型在COCO VAL 2017数据集上，IOU 阈值为 0.50:0.95 时的精度值为39.8%，而原模型的精度是40%，和原模型相差不到1%，所以满足精度要求。



## 6 常见问题

### 6.1 检测输出帧率过低问题

**问题描述：**

控制台输出检测的帧率严重低于25fps（如下10fps）。

```bash
I20220727 09:21:02.990229 32360 MxpiVideoEncoder.cpp:324] Plugin(mxpi_videoencoder0) fps (10).
```

**解决方案：**

确保输入的视频帧率高于25fps。

### 6.2 视频编码参数配置错误问题

**问题描述：**

运行过程中报错如下：

```bash
E20220728 17:05:59.947093 19710 DvppWrapper.cpp:573] input width(888) is not same as venc input format(1280)
E20220728 17:05:59.947126 19710 MxpiVideoEncoder.cpp:310] [mxpi_videoencoder0][2010][DVPP: encode H264 or H265 fail] Encode fail.
```

`pipeline/rtmOpenpose.pipeline`中视频编码插件分辨率参数指定错误。手动指定imageHeight 和 imageWidth 属性，需要和视频输入分配率相同。

**解决方案：**

确保`pipeline/rtmOpenpose.pipeline`中 mxpi_videoencoder0 插件中的 imageHeight 和 imageWidth 为输入视频的实际高和宽。

### 6.3 pth模型转onnx模型报错

**问题描述：**

在pth模型转onnx模型的过程中，执行 `convert_to_onnx.py` 脚本时报如下错误：

```
Traceback (most recent call last):
  File "/home/wxg/zhongzhi/RTMHumanKeypointsDetection/lightweight-human-pose-estimation.pytorch/scripts/convert_to_onnx.py", line 26, in <module>
    checkpoint = torch.load(args.checkpoint_path)
  File "/usr/local/python3.9.2/lib/python3.9/site-packages/torch/serialization.py", line 795, in load
    return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
  File "/usr/local/python3.9.2/lib/python3.9/site-packages/torch/serialization.py", line 1012, in _legacy_load
    result = unpickler.load()
  File "/usr/local/python3.9.2/lib/python3.9/site-packages/torch/serialization.py", line 958, in persistent_load
    wrap_storage=restore_location(obj, location),
  File "/usr/local/python3.9.2/lib/python3.9/site-packages/torch/serialization.py", line 215, in default_restore_location
    result = fn(storage, location)
  File "/usr/local/python3.9.2/lib/python3.9/site-packages/torch/serialization.py", line 182, in _cuda_deserialize
    device = validate_cuda_device(location)
  File "/usr/local/python3.9.2/lib/python3.9/site-packages/torch/serialization.py", line 166, in validate_cuda_device
    raise RuntimeError('Attempting to deserialize object on a CUDA '
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
```

**解决方案：**

pth模型转onnx模型调用了“CUDA”，此依赖操作需要基于“GPU”卡，昇腾服务器只存在“NPU”卡。若想正常转换onnx模型，需修改“/usr/local/python3.9.2/lib/python3.9/site-packages/torch/serialization.py”脚本655行的“MAP_LOCATION” 指定“cpu”进行pth模型转换onnx模型。
