## SDK 图像检测样例运行

### 介绍
提供的demo，实现图像检测样例运行并且输出检测结果写入图片result.jpg

### 准备工作
将样例目录python 从mxsdkreferenceapps/utorials/ImageDetectionSample文件夹下 移动到${SDK安装路径}/mxVision/samples/mxVision/python/路径下

可以使用mv 命令
进入到移动后的工程路径下

### 模型转换
参看图像检测样例运行C++ 章节转换模型
进入工程 model 文件夹 运行，模型转换脚本model_conversion.sh
```
bash model_conversion.sh
```

### pipeline 准备
将main.py 文件中 mxpi_objectpostprocessor0插件中的postProcessLibPath路径中的${SDK安装路径} 替换为自己的SDK安装路径

### 配置环境变量
```
. ${ascend_toolkit_path}/set_env.sh
. ${SDK-path}/set_env.sh

# 环境变量介绍
SDK-path: SDK mxVision 安装路径
ascend-toolkit-path: CANN 安装路径
```

### 运行
准备一张待检测图片，放到项目目录下命名为test.jpg
命令行输入：
python3 main.py

```
python3 main.py
```

### 查看结果
结果图片有画框，框的左上角显示推理结果和对应的confidence