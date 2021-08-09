# C++ 基于MxBase 的人群计数图像检测样例及后处理模块开发

##  介绍

本开发样例是基于mxBase开发的端到端推理的C++应用程序，可在昇腾芯片上进行 人群计数目标检测，并把可视化结果保存到本地。其中包含人群计数的后处理模块开发。 该Sample的主要处理流程为： Init > ReadImage >Resize > Inference >PostProcess >DeInit

## 模型转换

**步骤1** 

下载原始模型权重下载：

[https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/count_person.caffe.caffemodel](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC Model/crowdCount/count_person.caffe.caffemodel)

**步骤2** 

下载原始模型网络：

[https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/count_person.caffe.prototxt](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC Model/crowdCount/count_person.caffe.prototxt)

**步骤3**

下载对应的cfg文件：

[https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/crowdCount/insert_op.cfg](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC Model/crowdCount/insert_op.cfg)

**步骤4**

配置环境变量：

```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --input_shape="blob1:1,3,800,1408" --weight="count_person.caffe.caffemodel" --input_format=NCHW --output="count_person.caffe" --soc_version=Ascend310 --insert_op_conf=insert_op.cfg --framework=0 --model="count_person.caffe.prototxt" 
```

##  编译与运行

**步骤1** 

修改CMakeLists.txt文件 将set(MX_SDK_HOME SDK安装路径)中的SDK安装路径)中的{SDK安装路径}替换为实际的SDK安装路径

**步骤2**

设置环境变量 ASCEND_HOME Ascend安装的路径，一般为/usr/local/Ascend LD_LIBRARY_PATH 指定程序运行时依赖的动态库查找路径，包括ACL，开源软件库，libmxbase.so以及libyolov3postprocess.so的路径

```
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
```

**步骤3**

cd到CrowdCounting目录下，执行如下编译命令： bash build.sh

**步骤4**

下载人群计数的图像：

```
wget https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/crowdCount/crowd.jpg
```

准备一张推理图片放入CrowdCounting目录下，执行：

```
./mxBase_sample ./crowd.jpg
```