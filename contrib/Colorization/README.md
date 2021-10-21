# 黑白图像上色

## 1 介绍

在智能手机越来越普及的今天，拍摄一张色彩鲜艳、清晰的照片轻而易举。但是老照片没有如此“幸运”，大多为黑白。借助人工智能，可以一定程度上帮助老照片还原原来色彩。

本项目是黑白图像上色应用，旨在华为Atlas300推理芯片上实现输入黑白图像，自动对黑白图像进行上色，还原彩色图像。

### 1.1 支持的产品

Atlas300推理芯片

### 1.2 代码目录结构与说明

.
├── data
├── model
│   ├── colorization.caffemodel
│   └── colorization.prototxt
├── out
├── pipeline
│   └── colorization.pipeline
├── README.md
├── scripts
│   ├── atc_run.sh
│   └── run.sh
└── src
    └── main.py

## 2 环境依赖

### 2.1 环境变量

模型转换和工程运行的环境变量已写入对应的shell脚本中

### 2.2 软件依赖

|     依赖软件     | 版本  |
|------------------|-------| 
|     python       | 3.7.5 | 
|    MindX_SDK     | 2.0.2 |
|   opencv-python  | 4.5.3 |
|      numpy       | 1.21.2|  

## 3 运行

示例步骤如下：
### 3.1 模型转换

本工程原模型是caffee模型，需要使用atc工具转换为om模型，模型和所需权重文件已随工程上传，如需自行下载请使用以下命令下载

```
wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization.prototxt
wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/colorization/colorization.caffemodel
```

下载完毕后将模型和权重文件移动至model目录，后进入scripts目录执行模型转换脚本

```
cd scripts
bash atc_run.sh
```

### 3.2 获取测试图片

将待上色图片移动至data目录。本样例使用图片方式获取如下

```
cd data
wget https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/colorization_picture-python/dog.png
```

### 3.3 运行推理工程

进入scripts目录，修改INPUT_PIC变量为输入图片的路径，本示例为"../data/dog.png"，修改MX_SDK_HOME环境变量为SDK安装路径。
执行脚本

```
bash run.sh
```

输出结果保存在out目录下，下载至本地查看图片上色是否合理


