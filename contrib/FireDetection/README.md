# 高速公路车辆火灾识别

## 1 介绍

高速公路车辆火灾识别基于 MindX SDK 开发，在 Ascend 310 芯片上进行目标检测，将检测结果保存成图片。项目主要流程为：通过 live555 服务器进行拉流输入视频，然后进行视频解码将 H.264 格式的视频解码为图片，图片缩放后经过模型推理进行火焰和烟雾检测，识别结果经过后处理后利用 cv 可视化识别框，如果检测到烟雾和火灾进行告警。

### 1.1 支持的产品

昇腾 310（推理）

### 1.2 支持的版本

本样例配套的 CANN 版本为 [5.0.4](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2Fcann%2Fcommercial)，MindX SDK 版本为 [2.0.4](https://www.hiascend.com/software/Mindx-sdk)。

MindX SDK 安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 软件方案介绍

基于 MindX SDK 的高速公路车辆火灾识别业务流程为：将待检测的视频放在 live555 服务器上经`mxpi_rtspsrc`拉流插件输入，然后使用视频解码插件`mxpi_videodecoder`将视频解码成图片，再通过图像缩放插件`mxpi_imageresize`将图像缩放至满足检测模型要求的输入图像大小要求，缩放后的图像输入模型推理插件`mxpi_modelinfer`得到检测结果，如果发生检测到火焰或者烟雾报警并保存该帧图片便于查看。

### 1.4 代码目录结构与说明

本 Sample 工程名称为 **FireDetection**，工程目录如下图所示：

```
├── envs
│   └── env.sh                   //基础环境变量
├── images                       //ReadMe图片资源
│   └── image-flow.png
├── model
│   ├── aipp_yolov5.cfg          //atc转换时需要的aipp配置文件
│   ├── model_conver.sh                   //模型转换脚本
│   ├── modify_yolov5s_slice.py  //slice算子处理脚本
│   ├── yolov5.cfg               //om模型后处理配置文件
│   └── yolov5.names             //om模型识别类别文件
├── pipeline
│   ├── fire_p.pipeline          //test使用的pipeline文件
│   └── fire_v.pipeline          //视频流使用的pipeline文件
├── test
│   ├── map_calculate.py         //mAP计算(精度计算)
│   ├── parse_COCO.py            //利用标记生成真实标签
│   └── testmain.py              //识别，保存结果，并进行性能测试
├── result                       //运行结果图片保存位置
├── main.py                      //识别streams脚本
├── run.sh                       //运行脚本
└── README.md
```

###  1.5 技术实现流程图

![image-20211209173032019](images/image-flow.png)

## 2 环境依赖

| 软件名称            | 版本        | 说明                          | 获取方式                                                     |
| ------------------- | ----------- | ----------------------------- | ------------------------------------------------------------ |
| MindX SDK           | 2.0.4       | mxVision软件包                | [链接](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2Fmindx-sdk%2Fmxvision) |
| ubuntu              | 18.04.1 LTS | 操作系统                      | Ubuntu官网获取                                               |
| Ascend-CANN-toolkit | 5.0.4       | Ascend-cann-toolkit开发套件包 | [链接](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2Fcann%2Fcommercial) |

在运行项目需要的环境变量如下，运行前不需要特别设置，环境依赖已经写入脚本中，脚本在`FireDetection/envs`目录下：

```bash
# 基础环境变量——env.sh
export MX_SDK_HOME="${SDK安装路径}/mxVision"
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${MX_SDK_HOME}/python:${PYTHONPATH}"

```

注：其中`${SDK安装路径}`替换为用户的SDK安装路径;`install_path`替换为ascend-toolkit开发套件包所在路径。`LD_LIBRARY_PATH`用以加载开发套件包中lib库。

## 3 模型转换以及依赖安装

本项目使用的模型是火灾识别的模型。模型文件可以直接下载。

### 3.1 模型转换

使用模型转换工具 ATC 将 onnx 模型转换为 om 模型，模型转换工具相关介绍参考链接：[CANN 社区版]([前言_昇腾CANN社区版(5.0.4.alpha002)(推理)_ATC模型转换_华为云 (huaweicloud.com)](https://support.huaweicloud.com/atctool-cann504alpha2infer/atlasatc_16_0001.html)) 。

步骤如下：

- **步骤1** 下载原开源项目并在原项目中将`pt`文件转换成`onnx`模型，也可以下载`onnx`模型，请移动至`FireDetection/model`目录下，并跳至**步骤5**；若下载`om`模型文件，请跳过模型转换步骤。

- **步骤2** 在原开源项目中将`pt`文件转换为`onnx`文件，在`fire-smoke-detect-yolov4\yolov5`目录下执行

  ```bash
  python3.9.2 /models/export.py --weights ./best.pt
  ```

  运行结果：生成`best.onnx`文件。

- **步骤3** `onnx`文件简化

  对导出的`onnx`图使用`onnx-simplifer`工具进行简化。运行如下命令：

  ```bash
  python3.9.2 -m onnxsim --skip-optimization best.onnx best_s.onnx
  ```

  运行结果：生成`best_s.onnx`文件。

- **步骤4** Slice算子处理

  移动上一步生成的`best_s.onnx`文件至`FireDetection\model`，并使用该目录下的脚本`modify_yolov5s_slice.py`修改模型Slice算子

  ```bash
  python3.9.2 modify_yolov5s_slice.py best_s.onnx
  ```

  运行结果：生成`best_s_t.onnx`文件。

- **步骤5** 将`best_s_t.onnx`文件重命名为`firedetection.onnx`，然后运行当前目录`FireDetection\model`下的`model_conver.sh`

  ```bash
  bash model_conver.sh
  ```

  执行该命令后会在当前文件夹下生成项目需要的模型文件

  ```
  ATC start working now, please wait for a moment.
  ATC run success, welcome to the next use.
  ```

  表示命令执行成功。

###  3.2 准备

按照第3小节**软件依赖**安装 live555 和 ffmpeg，按照 [Live555离线视频转RTSP说明文档](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/Live555离线视频转RTSP说明文档.md)将 mp4 视频转换为 H.264 格式。并将生成的 H.264 格式的视频上传到`live/mediaServer`目录下，然后修改`pipeline`目录下的`fire_v.pipeline`文件中`mxpi_rtspsrc0`的内容。

```json
"mxpi_rtspsrc0": {
	"props": {
		"rtspUrl":"rtsp://xxx.xxx.xxx.xxx:xxxx/xxx.264",  // 修改为自己所使用的的服务器和文件名
	},
    "factory": "mxpi_rtspsrc",
	"next": "mxpi_videodecoder0"
}
```

##  4 运行与测试

### 4.1 运行

- **步骤1** 按照第 2 小节 **环境依赖** 中的步骤设置环境变量。

- **步骤2** 按照第 3 小节 **模型转换** 中的步骤获得 `om` 模型文件，放置在 `FireDetection/model` 目录下。
- **步骤3** 修改`fire_v.pipline`中`mxpi_modelinfer0`中`postProcessLibPath`的值`${SDK安装路径}`为 MindX SDK 的安装路径

- **步骤4** 运行。进入 `FireDetection` 目录，在 `FireDetection` 目录下执行命令：

```bash
bash run.sh
```

运行结果会保存在`FireDetection/result`目录下

### 4.2 性能与精度测试

- **步骤1** 下载测试数据集，并将`JPEGImages`目录以及`test.json`放在`FireDetection/test`目录下，并在`FireDetection/test`目录下运行指令

  ![dataset](images/dataset.png)

  ```bash
  python3.9.2 parse_COCO.py
  ```

  运行脚本后会自动生成真实标签在`ground-truth`路径下以`.txt`格式保存。

- **步骤2** 在`FireDetection/test`目录下运行`testmain.py`脚本，对`JPEGImages`目录下的图片进行识别并输出结果

  ```bash
  mkdir test_result
  python3.9.2 testmain.py
  ```

  运行脚本后会生成经过 SDK 后的推理结果结果保留在`test-result`目录下以`.txt`格式保存。

  运行结果中会有`Spend time：`是识别所有图片所用的时间，`fps：`计算得到的帧数

  ![fps](images/fps.png)

- **步骤3** 在`FireDetection/test`目录下运行`map_calculate.py`脚本，计算精度。

  ```bash
  python3.9.2 map_calculate.py
  ```

  测试结果

  ![ap](images/ap.png)

  ##  5 软件依赖以及资源链接

  推理中涉及到第三方软件依赖如下表所示。

  | 依赖软件       | 版本       | 说明                                       | 使用教程                                                     |
  | -------------- | ---------- | ------------------------------------------ | ------------------------------------------------------------ |
  | live555        | 1.09       | 实现视频转 rstp 进行推流                   | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/Live555离线视频转RTSP说明文档.md) |
  | ffmpeg         | 2021-10-14 | 实现 mp4 格式视频转为 H.264 格式视频       | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/pc端ffmpeg安装教程.md#https://gitee.com/link?target=https%3A%2F%2Fffmpeg.org%2Fdownload.html) |
  | pytorch        | 1.7.1      | conda install pytorch                      | python 3.9.2                                                 |
  | torchvision    | 0.8.2      | conda install torchvision                  | python 3.9.2                                                 |
  | onnx           | 1.60       | conda install onnx                         | python 3.9.2                                                 |
  | onnx-simplifer | 0.3.6      | python3.9.2 -m pip install onnx-simplifier | [链接](https://github.com/daquexian/onnx-simplifier)         |
  | 模型文件       | -          | pt 模型文件，onnx 模型文件，om 模型文件    | [链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/FireDetection/models.zip) |
  | 测试数据集     |            | 由1946张图片构成，COCO 格式                | [链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/FireDetection/dataset.zip) |
  | 原项目链接     | -          | -                                          | [链接](https://github.com/gengyanlei/fire-smoke-detect-yolov4) |