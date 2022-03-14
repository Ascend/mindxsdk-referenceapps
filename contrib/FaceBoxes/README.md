# FaceBoxes

## 1 介绍

本开发项目演示Faceboxes模型实现人脸检测。本系统基于mxVision SDK进行开发，以昇腾Atlas300卡为主要的硬件平台，主要应用于在CPU上实现实时的人脸检测，检测图像中的场景中人脸不能被遮挡严重且亮度过低，并且不能有与人脸形态相似的动物出现。项目的主要流程为：

1.环境搭建；
2.模型转换；
3.数据集预处理;
4.模型离线推理；
5.精度、性能对比


## 2 环境依赖

* 支持的硬件形态和操作系统版本

  | 硬件形态                              | 操作系统版本   |
  | ------------------------------------- | -------------- |
  | x86_64+Atlas 300I 推理卡（型号3010）  | Ubuntu 18.04.1 |
  | x86_64+Atlas 300I 推理卡 （型号3010） | CentOS 7.6     |
  | ARM+Atlas 300I 推理卡 （型号3000）    | Ubuntu 18.04.1 |
  | ARM+Atlas 300I 推理卡 （型号3000）    | CentOS 7.6     |

* 软件依赖

  | 软件名称 | 版本  |
  | -------- | ----- |
  | cmake    | 3.5.+ |
  | mxVision | 2.0.2 |
  | Python   | 3.9.2 |
  | Pytorch   | 1.9.0 |
  | CANN   | 3.3.0 |
  | OpenCV   | 4.5.3 |
  | gcc      | 7.5.0 |
  | ffmpeg   | 3.4.8 |

## ３ 代码主要目录介绍

本项目工程名称为FaceBoxes，工程目录如下图所示：

```
.
├── data
│   ├── FDDB
│   │   ├── images
│   │   │	 ├── ...
│   │   └── img_list
│   └── FDDB_Evaluation
│   │   ├── FDDB_dets.txt
│   └── results
│   │   ├── ...
│   └── ground_truth
│   │   ├── ...
│   └── pred_sample
│   │   ├── 1
│   │   │	 ├── ...
│   └── 1
├── weights
│   ├── FaceBoxesProd.pth
├── models
│   ├── faceboxes-b0_bs1.om
│   ├── faceboxes-b0_bs1.onnx
├── pipeline
│   ├── Faceboxes.pipeline
├── plugins
│   ├── FaceBoxesPostProcess // 后处理插件
│   │   ├── CMakeLists.txt
│   │   ├── FaceBoxesPostProcess.cpp
│   │   ├── FaceBoxesPostProcess.h
│   │   └── build.sh
├── script
│   ├── convert.py
│   ├── split.py
│   ├── evaluate.py
│   ├── setup.py
│   ├── box_overlaps.pyx
├── main.py
├── test.py
├── README.md
├── build.sh
└── run.sh
```

## 4 软件方案介绍

为了完成图像中的人脸检测，系统需要检测出图像中的人脸，因此系统中需要包含人脸检测、后处理、可视化。其中人脸检测模块选择Faceboxes，得到人脸预测框的参数和置信度；后处理模块进行预测框的解码和非极大值抑制处理，输出置信度最高的人脸预测框坐标参数及其置信度；可视化模块根据输出结果对图像中的人脸进行画框并且标注置信度。系统方案中各模块功能如表1.1 所示。

表1.1 系统方案中个模块功能：

| 序号 | 子系统         | 功能描述                                                     |
| ---- | -------------- | ------------------------------------------------------------ |
| 1    | 初始化配置     | 主要用于初始化资源，如线程数量、共享内存等。                 |
| 2    | 图像预处理     | 在进行基于深度神经网络的图像处理之前，需要将图像缩放到固定的尺寸和格式。 |
| 3    | 人脸检测       | 基于深度学习的人脸检测算法是该系统的核心模块之一，本方案选用基于Faceboxes的人脸检测。 |
| 4    | 后处理       | Faceboxes的输出为预测框相对于先验框的偏移量，所以首先需要生成先验框并且根据偏移量得出预测框的坐标参数，然后对所有预测框进行nms处理。 |
| 5    | 可视化       | 对图像中的人脸进行画框并且标注置信度。 |

## 5 准备

**步骤1：** 参考安装教程《mxVision 用户指南》安装 mxVision SDK。

**步骤2：** 配置 mxVision SDK 环境变量。

`export MX_SDK_HOME=${安装路径}/mxVision `

注：本例中mxVision SDK安装路径为 /home/uestc_luo1/MindX_SDK/mxVision。

**步骤3：** 推荐在${MX_SDK_HOME}/samples/mxVision下创建FaceBoxes根目录，在项目根目录下创建目录models `mkdir models`，将离线模型faceboxes-b0_bs1.om文件放入文件夹下。

**步骤4：** 编译程序前提需要先交叉编译好第三方依赖库。

**步骤5：** 配置环境变量MX_SDK_HOME：

```bash
export MX_SDK_HOME=/MindX_SDK/mxVision/								
# 此处MX_SDK_HOME请使用MindX_SDK的实际路径
```

**步骤6**：在插件代码目录下创建build文件夹，使用cmake命令进行编译，生成.so文件。下面以单人独处插件的编译过程作为范例：

```bash
## 进入目录 /plugin
## 创建build目录
mkdir build
## 使用cmake命令进行编译
cmake ..
make -j
make install
```

编译好的插件会自动存放到SDK的插件库中，可以直接在pipeline中使用。

**步骤7:** 配置pipeline

1.  插件参数介绍

   * mxpi_objectpostprocessor0
   
     | 参数名称        | 参数解释              |
     | :-------------- | --------------------- |
     | dataSource     | 输入数据对应索引（通常情况下为上游元件名称），默认为上游插件对应输出端口的key值 |
     | postProcessConfigPath  | 后处理配置文件路径 |
     | postProcessLibPath    | 后处理配置     |
 
2. 配置范例

   ```
   ## /*Faceboxes*/
        "mxpi_tensorinfer0": {
            "props": {
            "modelPath": "./models/faceboxes-b0_bs1.om"
        },
        "factory": "mxpi_tensorinfer",
        "next": "mxpi_objectpostprocessor0"
        },
        "mxpi_objectpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "./config/faceboxes-b0_bs1.cfg",
                "postProcessLibPath": "${MX_SDK_HOME}/lib/modelpostprocessors/libfaceboxespostprocess.so"
        },
        "factory": "mxpi_objectpostprocessor",
        "next": "mxpi_dataserialize0"
        },
 ```
   根据所需场景，配置pipeline文件，调整路径参数以及插件阈值参数。例如“postProcessLibPath”字段是SDK模型后处理插件库路径。


**步骤8：** 在main.py中，修改pipeline路径、对应的流名称以及需要获取结果的插件名称，最终程序运行完将所有图片的可视化结果写在data/results目录下，参数结果写在data/FDDB_Evaluation/FDDB_dets.txt里

```python
## 插件位置
with open("./pipeline/Faceboxes.pipeline", 'rb') as f:
        dataInput.data = f.read()
## pipeline中的流名称
streamName = b'Faceboxes'
## 想要获取结果的插件名称
key = b'mxpi_objectpostprocessor0'
```
## 6 模型转换

本项目中用到的模型有：Faceboxes

### 6.1 pth转onnx模型

1.FaceBoxes模型代码下载
```
cd ./FaceBoxes
git clone https://github.com/zisianw/FaceBoxes.PyTorch.git
```
2.预训练模型获取。
```
到以下链接下载原预训练模型和onnx模型文件，分别放在/weights和/models 目录下：
(https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/Faceboxes/model.zip) 

### 6.2 onnx转om模型

1.设置环境变量
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest

export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH

export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH

export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH

export ASCEND_OPP_PATH=${install_path}/opp
```
2.在models目录下，使用atc将onnx模型转换为om模型文件，加入--insert_op_conf参数使用AIPP，放到models目录下，工具使用方法可以参考CANN 5.0.2 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=faceboxes-b0_bs1.onnx --output=faceboxes-b0_bs1 --input_format=NCHW --input_shape="image:1,3,1024,1024" --log=debug --soc_version=Ascend310 --insert_op_conf=../config/FaceBoxes.aippconfig

## 7 测试

准备好要测试的图片，在test.py中修改好测试图片读取路径以及结果存放路径，并修改 run.sh 文件中的环境路径和项目路径以及要运行的python文件名test.py。

```bash
export MX_SDK_HOME=${CUR_PATH}/../../..
## 注意当前目录CUR_PATH与MX_SDK_HOME环境目录的相对位置
```
直接运行

```bash
chmod +x run.sh
./run.sh
```


## 8 数据集获取

该模型使用[FDDB官网]的2845张验证集进行测试，图片与标签分别存放在/data/FDDB/images与/data/FDDB/img_list.txt，链接为：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/Faceboxes/data.zip

## 9 精度验证

### 9.1 准备
1.下载FDDB数据集注释，链接为：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/Faceboxes/data.zip，将注释放在data/ground_truth下面

2.下载所需脚本box_overlaps.pyx、setup.py、convert.py、split.py、evaluate.py，放到script目录下，下载链接为：https://gitee.com/ascend/modelzoo/tree/master/contrib/ACL_PyTorch/Research/cv/face/FaceBoxes/FDDB_Evaluation

3.将split.py与evaluate.py的代码进行修改：
（1）split.py: 
在第22行代码后添加：
cur_path_dir = os.path.join(cur_path, '1')
if not os.path.exists(cur_path_dir):
    os.makedirs(cur_path_dir)
（2）evaluate.py: 
在第229行与第230行之间添加：for fold_id in range(1, 11)，添加后代码如下：
# different setting
for fold_id in range(1, 11):
            count_face = 0
            ...

在第233行代码上进行修改：
gt = gt_box_dict[event[setting_id]] → gt = gt_box_dict[fold_id]

### 9.2 验证流程
在运行完main.py后开始进行精度验证，所需代码文件放在script目录下。首先将所依赖的python包安装好（bbox除外），bbox函数直接在终端运行python3.7 setup.py install即可。之后分别运行script目录下的convert.py，split.py和evaluate.py，FDDB集的精度结果在运行完evaluate.py后会打印出来。


## 10 常见问题
### 10.1 模型路径配置

#### 问题描述：

检测过程中用到的模型以及模型后处理插件需配置路径属性。

#### 后处理插件配置范例：

```json
        "mxpi_objectpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "./config/faceboxes-b0_bs1.cfg",
                "postProcessLibPath": "${MX_SDK_HOME}/lib/modelpostprocessors/libfaceboxespostprocess.so"
        },
        "factory": "mxpi_objectpostprocessor",
        "next": "mxpi_dataserialize0"
        }
```


