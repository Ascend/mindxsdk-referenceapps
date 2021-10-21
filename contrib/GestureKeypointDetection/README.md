## 基于MindX SDK开发目标检测应用

## 简介

样例实现图像目标检测功能。具体流程为`图像解码`->`图像缩放`->`模型推理`->`模型后处理`。

## 运行步骤

- **安装python的opencv**

```bash
pip3.7 install opencv-python -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```

- **模型准备**

上一周转换的yolov3模型，拷贝到`model`目录下

- **配置环境变量**

```bash
export MX_SDK_HOME=/home/HwHiAiUser/MindX/MindXSDK/mxVision
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
export PYTHONPATH=${MX_SDK_HOME}/python
```

- **运行**

```bash
python3 main.py
```

