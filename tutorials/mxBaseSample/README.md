
# C++ 基于MxBase 的yolov3图像检测样例及yolov3的后处理模块开发

## 介绍
本开发样例是基于mxBase开发的端到端推理的C++应用程序，可在昇腾芯片上进行 yolov3 目标检测，并把可视化结果保存到本地。其中包含yolov3的后处理模块开发。
该Sample的主要处理流程为：
Init > ReadImage >Resize > Inference >PostProcess >DeInit

## 模型转换

**步骤1** 模型获取
在ModelZoo上下载YOLOv3模型 ，选择“历史版本”中版本1.1下载。[下载地址](https://www.hiascend.com/zh/software/modelzoo/detail/C/210261e64adc42d2b3d84c447844e4c7)
**步骤2** 模型存放
将获取到的YOLOv3模型pb文件放至上一级的models文件夹中
**步骤3** 执行模型转换命令

(1) 配置环境变量
#### 设置环境变量（请确认install_path路径是否正确）
#### Set environment PATH (Please confirm that the install_path is correct).
```c
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

```
(2) 转换模型
```
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input/input_data:1,416,416,3" --out_nodes="conv_lbbox/BiasAdd:0;conv_mbbox/BiasAdd:0;conv_sbbox/BiasAdd:0"
```

## 编译与运行
**步骤1** 修改CMakeLists.txt文件 将set(MX_SDK_HOME ${SDK安装路径}) 中的${SDK安装路径}替换为实际的SDK安装路径

**步骤2** 设置环境变量
ASCEND_HOME Ascend安装的路径，一般为/usr/local/Ascend
LD_LIBRARY_PATH 指定程序运行时依赖的动态库查找路径，包括ACL，开源软件库，libmxbase.so以及libyolov3postprocess.so的路径
```
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
```

**步骤3** cd到mxbase目录下，执行如下编译命令：
bash build.sh

**步骤4** 制定jpg图片进行推理，准备一张推理图片放入mxbase 目录下。eg:推理图片为test.jpg
cd 到mxbase 目录下
```
./mxBase_sample ./test.jpg
```
