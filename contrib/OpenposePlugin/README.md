# 基于 Openpose 和 MinxXSDK 的人体关键点插件开发
## 1. 简介
本开发插件基于 MindXSDK 开发，在晟腾芯片上进行人体关键点和骨架检测，将检测结果可视化并保存。输入一幅图像，可以检测得到图像中所有行人的关键点并连接成人体骨架。

本方案中，采用 openpose 模型提取人体关键点，然后通过人体姿态描绘插件连接关键点形成人体姿态描绘图，在 MindXSDK 平台上的实现流程为：输入图片类型是图片数据（jpg图片序列），通过调用MindX SDK提供的图像解码接口，解码后获取图像数据，然后经过图像尺寸大小变换，满足检测模型要求的输入图像大小要求；将尺寸变换后的图像数据输入人体关键点检测模型进行推理，模型输出经过后处理后，得到人体关键点和关键点之间的连接关系，输出关键点连接形成的人体姿态描绘图。除模型后处理插件外，其余插件已经具备，可以直接使用，本方案完成模型后处理插件开发，在模型后处理插件中对模型输出进行计算，进行关键点去重、骨架排序筛选、组成人体等后处理操作步骤，得到最终的检测结果。


## 2. 代码目录结构与说明

├── mindx_sdk_plugin # Openpose 后处理插件目录
│   ├── CMakeLists.txt
│   ├── MxpiOpenposePlugin.cpp
│   └── MxpiOpenposePlugin.h
├── proto # 后处理插件实现中自定义结构文件 
│   ├── CMakeLists.txt
│   └── mxpiOpenposeProto.proto
├── python
│   ├── test.jpg # 测试图片原图
│   ├── test_single_result.jpg # 测试图片结果
│   ├── main.py # 整体流程调用脚本
│   ├── models
│   │   └── model_conversion.sh # 模型转换脚本 
│   └── pipeline
│       └── OpenposePlugin.pipeline # 流程编排文件
└── README.MD




## 3. 环境配置和编译运行

### 3.1 环境配置

运行前设置环境变量
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

### 编译运行

1. 编译插件

（1）进入 proto 文件夹，执行如下命令
```
mkdir build
cd build
cmake ..
make -j
```
(2) 进入 mindx_sdk_plugin 文件夹，执行如下命令
```
mkdir build
cd build
make -j
cp libmxpi_openposeplugin.so ~/MindX_SDK/mxVision-2.0.2/lib/plugins/
```

2. 模型转换
参考链接：https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/%20gesturedetection/ATC_OpenPose_caffe_AE 中模型转换步骤，使用 python/models 文件夹下的模型转换脚本 model_conversion.sh 转换模型，转换过程中设置输入图片尺寸大小为 368, 368。将生成的 om 模型放到 python/models 文件夹中，名为 pose_deploy_368.om.


3. 运行
准备一张测试图片 test.jpg，进入 python 文件夹，运行 `python3.7 main.py`, 在当前文件夹下生成 test_single_result.jpg 结果图片，图片中标注出关键点和骨架。