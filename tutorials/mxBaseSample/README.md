
# C++ 基于MxBase 的yolov3图像检测样例及yolov3的后处理模块开发

## 介绍
本开发样例是基于mxBase开发的端到端推理的C++应用程序，可在昇腾芯片上进行 yolov3 目标检测，并把可视化结果保存到本地。其中包含yolov3的后处理模块开发。
该Sample的主要处理流程为：
Init > ReadImage >Resize > Inference >PostProcess >DeInit

## 模型转换

**步骤1** 模型获取
在ModelZoo上下载YOLOv3模型。[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ActionRecognition/ATC%20YOLOv3%28FP16%29%20from%20TensorFlow%20-%20Ascend310.zip)
**步骤2** 模型存放
将获取到的YOLOv3模型pb文件放至上一级的models文件夹中
**步骤3** 执行模型转换命令

(1) 配置环境变量
#### 设置toolkit环境变量（请确认install_path路径是否正确）
#### Set environment PATH (Please confirm that the install_path is correct).
```c
. /usr/local/Ascend/ascend-toolkit/set_env.sh # Ascend-cann-toolkit开发套件包默认安装路径，请根据实际安装路径修改。

```
(2) 转换模型(若运行在310B上，模型转换时需将Ascend310修改为Ascend310B1)
```
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"
```

## 编译与运行
**步骤1** 修改CMakeLists.txt文件 将set(MX_SDK_HOME ${SDK安装路径}) 中的${SDK安装路径}替换为实际的SDK安装路径

**步骤2** 设置mxVision环境变量
```
. ${MX_SDK_HOME}/set_env.sh # ${MX_SDK_HOME}替换为用户的SDK安装路径
```

**步骤3** cd到mxbase目录下，执行如下编译命令：
bash build.sh

**步骤4** 制定jpg图片进行推理，准备一张推理图片放入mxbase 目录下。eg:推理图片为test.jpg
cd 到mxbase 目录下
```
./mxBase_sample ./test.jpg
```
