# 头部姿态识别

## 1 介绍
在本系统中，目的是基于MindX SDK，在昇腾平台上，开发端到端头部姿态识别的参考设计，实现对图像中的头部进行姿态识别的功能，并把可视化结果保存到本地，达到功能要求。

样例输入：带有头部的jpg图片。

样例输出：头部上带有三位坐标轴确定头部姿态的jpg图片。

### 1.1 支持的产品

支持昇腾310芯片

### 1.2 支持的版本

支持的SDK版本，列出版本号查询方式。

版本号查询方法，在Atlas产品环境下，运行命令：

```
npu-smi info
```
可以查询支持SDK的版本号


### 1.3 软件方案介绍

本方案中，采用yolov4预训练模型对输入图片进行头部识别，头部识别后对识别出的头部图像进行抠图，然后使用WHENet模型进行头部姿态识别，最终根据WHENet模型识别得到的yaw，pitch，roll信息生成头部上带有三位坐标轴确定头部姿态的jpg图片。

注：由于YOLO模型限制，目前仅支持侧脸与正脸图片的头部姿态识别

### 1.4 代码目录结构与说明

本sample工程名称为HeadPoseEstimation，工程目录如下图所示：
```
├── models
│   ├── coco.names                  # 标签文件
│   ├── insert_op.cfg               # 模型转换aipp配置文件
│   ├── yolov4.cfg                  # yolo模型配置文件
│   ├── WHENet_b2_a1_modified.om    # 头部转换模型WHENet
│   └── yolov4_detection.om         # 头部识别模型YOLO
├── pipeline
│   └── recognition.pipeline        # pipeline文件
├── plugins
│   ├── MxpiHeadPoseEstimationPostProcess   # 姿态后处理插件
│   │   ├── CMakeLists.txt
│   │   ├── build.sh
│   │   ├── MxpiHeadPoseEstimationPostProcess.cpp
│   │   └── MxpiHeadPoseEstimationPostProcess.h
│   ├── MxpiHeadPoseEstimationPostProcess   # 自定义proto结构体
│   │   ├── CMakeLists.txt
│   │   ├── build.sh
│   │   └── mxpiHeadPoseProto.proto
│   └── build.sh
├── main.py
└── test.jpg
```

### 1.5 技术实现流程图

![diagram1](https://i.loli.net/2021/10/19/OqSelM4NZk6rtRd.jpg)

图1 头部姿态识别流程图

![diagram2](https://i.loli.net/2021/10/19/pmo81UAgzTS2QN4.jpg)

图2 头部姿态识别pipeline示意图

## 2 环境依赖

| 软件名称 | 版本   |
| :--------: | :------: |
|ubantu|18.04.1 LTS   |
|MindX SDK|2.0.2|
|Python|3.7.5|
|CANN|3.3.0|

在编译运行项目前，需要设置环境变量：

- 环境变量介绍

```
# 执行如下命令，打开.bashrc文件
vi .bashrc
# 在.bashrc文件中添加以下环境变量
MX_SDK_HOME=${SDK安装路径}

LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/

GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner

GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

#查看环境变量
env
```

## 3 模型获取

此处提供转换好的YOLOV4模型以及WHENet模型的om文件：[下载地址](https://pan.baidu.com/s/1E2FqL-X9zb0SM0v7BJu1SQ)，提取码：d5v9

注：**下载后请将两个模型请放置于models目录下**



## 4 编译与运行

示例步骤如下：

**步骤1** 

cd至`plugins/`    执行
```
bash build.sh
```
**步骤2** 

cd至`plugins/MxpiHeadPosePlugin/build/`   修改下面代码中的SDK目录并执行
```
cp libmxpi_headposeplugin.so {自己的MindX_SDK目录}/mxVision-2.0.2/lib/plugins/
```
**步骤3** 

修改`pipeline/recognition.pipeline`文件中: **mxpi_objectpostprocessor0**插件的`postProcessLibPath`属性，修改为
```
{SDK安装路径}/lib/modelpostprocessors/libyolov3postprocess.so
```
**步骤4** 

自行在网络找一张包含头部的jpg图像，重命名为test.jpg，放入项目根目录中，再执行
```
python3.7 main.py
```

