
# RCF模型边缘检测

## 1 介绍
本开发样例是基于mxBase开发的端到端推理的C++应用程序，可在昇腾芯片上进行 图像边缘提取，并把可视化结果保存到本地。
其中包含Rcf模型的后处理模块开发。 主要处理流程为： Init > ReadImage >Resize > Inference >PostProcess >DeInit

#### 1.1 支持的产品
昇腾310(推理)

#### 1.2 支持的版本
本样例配套的CANN版本为5.0.4，MindX SDK版本为2.0.4
MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1%E5%AE%89%E8%A3%85SDK%E5%BC%80%E5%8F%91%E5%A5%97%E4%BB%B6.md)

#### 1.3 代码目录结构与说明
本sample工程名称为EdgeDetectionPicture，工程目录如下图所示：

```
.
├── model
│   ├── aipp.cfg // 模型转换aipp配置文件
├── rcfDetection
│   ├── RcfDetection.cpp
│   └── RcfDetection.h
├──  rcfPostProcess
│   ├── rcfPostProcess.cpp
│   └── rcfPostProcess.h
├── build.sh
├── main.cpp
├── README.md
├── CMakeLists.txt
└── License
```

## 2 环境依赖
环境依赖软件和版本如下表：



| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 2.0.4        | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| Ascend-CANN-toolkit | 5.0.4        | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | Ubuntu 18.04 | 操作系统                      | Ubuntu官网获取                                               |

在编译运行项目前，需要设置环境变量：

- 环境变量介绍

  ```
  export install_path=/usr/local/Ascend/ascend-toolkit/latest
  export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  
  ```



## 3 模型转换

**步骤1** 模型获取
下载RCF模型 。[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/EdgeDetectionPicture/model.zip)

**步骤2** 模型存放
将获取到的RCF模型rcf.prototxt文件和rcf_bsds.caffemodel文件放在edge_detection_picture/model下

**步骤3** 执行模型转换命令

```
atc --model=rcf.prototxt --weight=./rcf_bsds.caffemodel --framework=0 --output=rcf --soc_version=Ascend310 --insert_op_conf=./aipp.cfg  --input_format=NCHW --output_type=FP32
```

## 4 编译与运行
**步骤1** 修改CMakeLists.txt文件 将set(MX_SDK_HOME ${SDK安装路径}) 中的${SDK安装路径}替换为实际的SDK安装路径

**步骤2** 设置环境变量
ASCEND_HOME Ascend安装的路径，一般为/usr/local/Ascend
LD_LIBRARY_PATH 指定程序运行时依赖的动态库查找路径，包括ACL，开源软件库，libmxbase.so的路径
```
export MX_SDK_HOME=${MX_SDK_HOME}
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
```

**步骤3** 执行如下编译命令：
bash build.sh

**步骤4** 进行图像边缘检测
请自行准备jpg格式的测试图像保存在文件夹中(例如 data/**.jpg)进行边缘检测 
```
./edge_detection_picture ./data
```
生成边缘检测图像 result/**.jpg

## 5 精度测试
下载开源数据集 BSDS500 [下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/EdgeDetectionPicture/data.zip), 使用 BSR/BSDS500/data/images/test数据进行测试


(1) 下载开源代码

``` shell
git clone https://github.com/Walstruzz/edge_eval_python.git
cd edge_eval_python
```
(2) 编译cxx

``` shell
cd cxx/src
source build.sh
```
(3) 将test中的图像经过边缘检测后的结果保存在result文件夹中

``` shell
./edge_detection_picture path/to/BSR/BSDS500/data/images/test/

```


(4) 修改检测代码

vim mian.py
注释第17行代码 nms_process(model_name_list, result_dir, save_dir, key, file_format)
修改18行为   eval_edge(alg, model_name_list, result_dir, gt_dir, workers)

vim /impl/edges_eval_dir.py
修改155行为  im = os.path.join(res_dir, "{}.jpg".format(i))

vim eval_edge.py
修改14行为  res_dir = result_dir

(5) 测试精度

``` shell
python main.py  --result_dir path/to/result  --gt_dir paht/to/BSR/BSDS500/data/groundTruth/test 

```
注: 
  result_dir: results directory

  gt_dir    : ground truth directory

