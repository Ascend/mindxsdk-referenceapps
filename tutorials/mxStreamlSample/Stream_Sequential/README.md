# 图像检测识别样例命令行运行

## 介绍

提供的sample样例,在代码里实现插件的编排，实现对本地图片进行YOLOv3+Resnet50目标检测。

###代码目录结构与说明
```
|-----data
|     |---images  //推理图片存放路径
|     |---models  //模型存放路径
|     |   |---resnet50
|     |   |---yolov3
|-----run.sh //编译运行脚本
|-----CMakeLists.txt
|-----main_sequential.cpp
```
### 环境依赖

|  软件名称        | 版本   |
|----------------- | ------|
|  mxVision        | 3.0.0 |
|  ascend-toolkit  | 5.0.4 |

### 依赖文件获取

**步骤1**  在路径：“SDK安装路径/mxVision-3.0.0/sample/mxVision/models” 下找到resnet50文件夹，拷贝至：“样例项目所在目录/data/models”下。
**步骤2**  在路径：“SDK安装路径/mxVision-3.0.0/sample/mxVision/models” 下找到yolov3文件夹，拷贝至：“样例项目所在目录/data/models”下。

### 模型获取

**步骤1**  在ModelZoo上下载YOLOv3模型。[下载地址](https://www.hiascend.com/zh/software/modelzoo/models/detail/C/210261e64adc42d2b3d84c447844e4c7/1)

**步骤2**  在ModelZoo上下载resnet50模型。[下载地址](https://www.hiascend.com/zh/software/modelzoo/models/detail/C/d63df55c1f7f4112a97c8a33e6da89fe/1)

**步骤3** 将获取到的模型的pb文件存放至：“样例项目所在目录/data/models/yolov3” 和 “样例项目所在目录/data/models/resnet50”。

### 模型转换
```
# 设置环境变量
在ascend-toolkit安装路径下执行命令：
source set_env.sh   //默认安装路径：usr/local/Ascend/ascend-toolkit

在SDK安装路径下执行命令：
source set_env.sh

# 在YOLOv3模型pb文件所在目录下执行命令：
atc --model=./yolov3_tf.pb --framework=3 --output=yolov3_tf_bs1_fp16 --output_type=FP32 --soc_version=Asscend310 --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0" --log=error --insert_op_conf=./aipp_yolov3_416_416.aippconfig

# 在resnet50模型pb文件所在目录下执行命令：
atc --model=./resnet50_tensorflow_1.7.pb --framework=3 --output=resnet50_aipp_tf --output_type=FP32 --soc_version=Asscend310 --input_shape="Placeholder:1,224,224,3" --input_format=NHWC --enable_small_channel=1 --log=error --insert_op_conf=./aipp_resnet50_224_224.aippconfig
```
执行完模型转换脚本后，会生成对应的.om模型文件。

### 编译运行

```
# 将命名为test的jpg图片放在/data/images路径下后，执行命令：
chmod +x run.sh
bash run.sh
```

### 输出结果

执行run.sh完毕后，打印图片目标识别和检测结果。

















