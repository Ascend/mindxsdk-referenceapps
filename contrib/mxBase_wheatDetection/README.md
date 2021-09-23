
# C++ 基于MxBase 的yolov5小麦检测

## 介绍
本开发样例是基于mxBase开发的端到端推理的小麦检测程序，实现对图像中的小麦进行识别检测的功能，并把可视化结果保存到本地。其中包含yolov5的后处理模块开发。
该Sample的主要处理流程为：
Init > ReadImage >Resize > Inference >PostProcess >DeInit

## 模型转换

**步骤1** 模型获取
在Kaggle上下载YOLOv5模型 。[下载地址](https://www.kaggle.com/yunyung/yolov5-wheat)
在Github上下载YOLOv5的各个文件。[下载地址](https://github.com/ultralytics/yolov5)
将下载的YOLOv5模型pt文件通过YOLOv5自带的export模型转换函数转换为onnx格式的文件
```
python export.py --weights best_v3.pt --img 416 --batch 1 --simplify
```
**步骤2** 模型存放
将获取到的YOLOv5模型onnx文件放至上一级的model文件夹中
**步骤3** 执行模型转换命令

(1) 配置环境变量
#### 设置环境变量（请确认install_path路径是否正确）
#### Set environment PATH (Please confirm that the install_path is correct).
```c
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

```
(2) 转换模型
```
atc --model=./best_v3_t.onnx --framework=5 --output=./onnx_best_v3 --soc_version=Ascend310 --insert_op_conf=./aipp.aippconfig --input_shape="images:1,3,416,416" --output_type="Conv_1228:0:FP32;Conv_1276:0:FP32;Conv_1324:0:FP32" --out_nodes="Conv_1228:0;Conv_1276:0;Conv_1324:0"
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

**步骤4** 制定jpg图片进行推理，将需要进行推理的图片放入mxbase 目录下的新文件夹中，例如mxbase/test。eg:推理图片为xxx.jpg
cd 到mxbase 目录下
```
./mxBase_sample ./test/
```
