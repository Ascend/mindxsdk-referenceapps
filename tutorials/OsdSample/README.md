# SampleOsd 样例说明

## 1 介绍
* 本样例从ExternalOsdInstances.json构建一个绘图单元集合（MxpiOsdInstancesList）的元数据（metadata）并送入stream
* 上传一张jpg格式图片并重命名为test.jpg，在运行目录下执行run.sh。请勿使用大分辨率图片
* 如构建的proto数据正确则可在程序运行结束后在运行目录找到图片testout.jpg，此图片为test.jpg经过缩放后加上绘图单元集合后的输出结果。

### 1.1 支持的产品
本项目以昇腾Atlas310、Atlas310B卡为主要硬件平台。

### 1.2 支持的版本
支持的SDK版本为 5.0.RC1, CANN 版本为 6.0.RC1。

 
## 2 环境依赖
在编译和运行项目需要的环境变量如下。

  ```
  export MX_SDK_path=""# mxVision 安装路径
  export Ascend_toolkit_path=""#CANN 安装路径
  
  # MindXSDK 环境变量：
  . /${MX_SDK_path}/set_env.sh
  
  # CANN 环境变量：
  . /${Ascend_toolkit_path}/set_env.sh
  ```

## 3 OSD模型转换
使用mxpi_opencvosd插件前，需要使用osd相关的模型文件，请执行mxVision安装目录下operators/opencvosd/generate_osd_om.sh脚本并生成所需的模型文件。

##  4 编译与运行
### 4.1 编译

```
bash build.sh
```

### 4.2运行
```
./main test.jpg
```