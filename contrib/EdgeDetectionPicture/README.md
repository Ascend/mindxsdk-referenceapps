
# C++ 基于MxBase rcf模型边缘检测以及rcf模型的后处理模块开发

## 介绍
本开发样例是基于mxBase开发的端到端推理的C++应用程序，可在昇腾芯片上进行 图像边缘提取，并把可视化结果保存到本地。
其中包含Rcf模型的后处理模块开发。 主要处理流程为： Init > ReadImage >Resize > Inference >PostProcess >DeInit

## 模型转换

**步骤1** 模型获取
下载RCF模型 。[下载地址](https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/edge_detection/ATC_RCF_Caffe_AE)

**步骤2** 模型存放
将获取到的RCF模型rcf.prototxt文件和rcf_bsds.caffemodel文件放在edge_detection_picture/model下

**步骤3** 执行模型转换命令

(1) 配置环境变量
#### 设置环境变量（请确认install_path路径是否正确）
#### Set environment PATH (Please confirm that the install_path is correct).
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

```
(2) 转换模型
```
atc --model=rcf.prototxt --weight=./rcf_bsds.caffemodel --framework=0 --output=rcf --soc_version=Ascend310 --insert_op_conf=./aipp.cfg  --input_format=NCHW --output_type=FP32
```

## 编译与运行
**步骤1** 修改CMakeLists.txt文件 将set(MX_SDK_HOME ${SDK安装路径}) 中的${SDK安装路径}替换为实际的SDK安装路径

**步骤2** 设置环境变量
ASCEND_HOME Ascend安装的路径，一般为/usr/local/Ascend
LD_LIBRARY_PATH 指定程序运行时依赖的动态库查找路径，包括ACL，开源软件库，libmxbase.so的路径
```
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
```

**步骤3** 执行如下编译命令：
bash build.sh

**步骤4** 进行图像边缘检测
请自行准备jpg格式的测试图像进行边缘检测 
```
./mxBase_sample ./**.jpg
```
生成边缘检测图像 result.jpg
