
# RCF模型边缘检测

## 1 介绍
本开发样例是基于mxBase开发的端到端推理的C++应用程序，可在昇腾芯片上进行 图像边缘提取，并把可视化结果保存到本地。
其中包含Rcf模型的后处理模块开发。 主要处理流程为： Init > ReadImage >Resize > Inference >PostProcess >DeInit

#### 1.1 支持的产品
昇腾310B1(推理)

#### 1.2 支持的版本
| 软件名称 | 版本   |
| -------- | ------ |
| python    | 3.9.2     | 
| MindX SDK     |    5.0RC1    |
| CANN | 310使用6.3.RC1<br>310B使用6.2.RC1 |

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

## 2 环境依赖                                            |

在编译运行项目前，需要设置环境变量：
```
. /usr/local/Ascend/ascend-toolkit/set_env.sh #toolkit默认安装路径，根据实际安装路径修改
. ${SDK_INSTALL_PATH}/mxVision/set_env.sh
```


## 3 模型转换

**步骤1** 模型获取
下载RCF模型 。[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/EdgeDetectionPicture/model.zip)

**步骤2** 模型存放
将获取到的RCF模型rcf.prototxt文件和rcf_bsds.caffemodel文件放在edge_detection_picture/model下

**步骤3** 执行模型转换命令

```
atc --model=rcf.prototxt --weight=./rcf_bsds.caffemodel --framework=0 --output=rcf --soc_version=Ascend310B1 --insert_op_conf=./aipp.cfg  --input_format=NCHW --output_type=FP32
```

## 4 编译与运行
**步骤1** 修改CMakeLists.txt文件 将set(MX_SDK_HOME ${SDK安装路径}) 中的${SDK安装路径}替换为实际的SDK安装路径

**步骤2** 执行如下编译命令：
bash build.sh

**步骤3** 进行图像边缘检测
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

