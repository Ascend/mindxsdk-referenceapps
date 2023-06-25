# 异步检测分类

## 1. 介绍
异步检测分类样例基于mxVision SDK进行开发，以310P为主要的硬件平台，支持基于图像的异步检测分类任务

### 1.1 支持的产品
310P
### 1.2 支持的版本
（软件包括第三方依赖）请列出环境依赖和版本
eg: 推荐系统为ubuntu 18.04 或者centos7.6，环境依赖软件和版本如下表：

| 软件名称     | 版本号    |
|----------|--------|
| cmake    | 3.10.2 |
| mxVision | 5.0RC2 |
| gcc      | 7.5.0  |


### 1.3 代码目录结构与说明
```
|-- AsyncInfer
|   |-- imgs_bak // 存放数据
|   |-- model
|   |   |-- resnet50  // 存放resnet50 om模型
|   |   |-- yolov3  // 存放yolo相关配置文件以及om模型
|-- main.cpp
|-- CMakeLists.txt
|-- README.MD
```

## 2. 模型下载

YoloV3模型下载：http://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/c-version/YoloV3_for_TensorFlow/zh/1.6/m/YOLOv3_TensorFlow_1.6_model.zip 

ResNet50模型下载: http://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/c-version/ResNet50_for_TensorFlow/zh/1.7/m/ResNet50_for_TensorFlow_1.7_model.zip

模型下载之后，放到model的对应目录下
## 3. 编译与运行

* 环境变量介绍
```shell
. ${sdk_path}/set_env.sh
. ${ascend_toolkit_path}/set_env.sh
```
* 在参考设计目录下创建build文件，使用cmake命令进行编译，生成.so文件，如下：

```bash
## 创建build目录
mkdir build
cd build

## 使用cmake命令进行编译
cmake ..
make -j
```
* 测试图片路径

测试图片放到./imgs_bak路径下


## 4.常见问题

请按照问题重要程度，详细列出可能遇到的问题和解决办法
