# 财务票据OCR识别

## 1 介绍
在本系统中，目的是基于MindX SDK，在昇腾平台上，开发端到端**财务票据OCR识别**的参考设计，实现**对财务票据中的文本信息进行OCR识别**的功能，达到功能要求。

样例输入：财务票据jpg图片

样例输出：框出主要文本信息并标记文本内容以及票据类型的jpg图片

注：由于数据集的限制，暂时只支持**增值税发票、出租车发票和定额发票**的识别



### 1.1 支持的产品

支持昇腾310芯片



### 1.2 支持的版本

支持21.0.4版本

版本号查询方法，在Atlas产品环境下，运行命令：

```
npu-smi info
```



### 1.3 软件方案介绍

本方案中，采用resnet50进行票据分类，采用DBNet识别文本框坐标信息，然后对文本信息进行抠图，送入CRNN模型进行文本识别，最后将分类结果、检测结果和识别结果可视化在输入图片上。



### 1.4 代码目录结构与说明

本sample工程名称为InvioceOCR，工程目录如下图所示：

```
├── lib                         # 后处理插件库 
│   ├── libclipper.so
│   └── libDBPostProcess.so
├── models
│   ├── crnn
│   │   ├── crnn.om
│   │   ├── crnn.onnx
│   │   ├── ppocr_keys_v1.names # 字典
│   │   ├── rec_aipp.cfg        # crnn-aipp转换配置文件
│   │   └── rec.cfg    			# crnn后处理配置
│   ├── db
│   │   ├── db.om
│   │   ├── db.onnx
│   │   ├── det_aipp.cfg
│   │   └── det.cfg 			# db后处理配置
│   └── resnet50
│       ├── aipp.config
│       ├── invoice.names		# 票据类别
│       ├── resnet50.air
│       ├── resnet50.cfg		# resnet50配置
│       └── resnet50.om
├── pipeline
│   └── InvoiceOCR.pipeline     # pipeline文件
├── inputs                      # 输入图片
│   └── xxx.jpg
├── outputs                     # 输出图片
│   └── xxxx.jpg
├── main.py                     # 推理脚本
├── eval.py						# 测试精度脚本
└── set_env.sh					# 环境变量设置脚本
```



### 1.5 技术实现流程图

![](imgs\技术流图.png)

图1 技术流程图

![](imgs\pipeline流程图.png)

图2 pipeline流程图



### 1.6 特性及适用场景

本项目适用于票据图片完整清晰，倾斜角度较小的场景，并且建议图片分辨率不超过1280*1280，大小不超过1M。

**注：**由于模型的限制，本项目暂只支持**增值税发票、出租车发票和定额发票**的识别



## 2 环境依赖

环境依赖软件和版本如下表：

| 软件名称  | 版本        |
| --------- | ----------- |
| ubantu    | 18.04.1 LTS |
| MindX SDK | 2.0.4       |
| Python    | 3.9.2       |
| CANN      | 5.0.4       |

在编译运行项目前，需要设置环境变量：

在项目路径下运行

```
bash set_env.sh
# 查看环境
env
```



## 3 模型获取、训练与转换

### 3.1 模型获取及训练

#### 3.1.1 resnet50

训练代码参考[链接](https://gitee.com/mindspore/models/tree/r1.8/official/cv/resnet)，将数据集放在.`/imagenet2012`目录下，修改config下配置文件，执行如下代码训练

```
python train.py --data_path=./imagenet2012/train --config_path=./config/resnet50_imagenet2012_config.yaml --output_path='./output' 
```

#### 3.1.2 db

参考[paddleocr文本检测训练](https://gitee.com/paddlepaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/detection.md)

#### 3.1.3 crnn

参考[paddleocr文本识别训练](https://gitee.com/paddlepaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/recognition.md)



### 3.2 模型转换

在`./models/resnet50`目录下执行如下命令

```
atc --model=./resnet50.air --framework=1 --output=resnet50 --input_format=NCHW --input_shape="x:1,3,224,224" --enable_small_channel=1 --soc_version=Ascend310 --insert_op_conf="aipp.config"
```

在`./models/db`目录下执行如下命令

```
atc --model=./db.onnx --framework=5 --output_type=FP32 --output=db_r50_1 --input_format=NCHW --input_shape="x:1,3,-1,-1" --dynamic_image_size="1216,1280;1280,1216;1120,1280;1280,1120;1024,1280;1280,1024;928,1280;1280,928;832,1280;1280,832;736,1280;1280,736;704,1280;1280,704;672,1280;1280,672;640,1280;1280,640;608,1280;1280,608;576,1280;1280,576;544,1280;1280,544;512,1280;1280,512;480,1280;1280,480;448,1280;1280,448" --soc_version=Ascend310 --insert_op_conf=./det_aipp.cfg
```

在`./models/crnn`目录下执行如下命令

```
atc --model=./crnn.onnx --framework=5 --output_type=FP32 --output=crnn --input_format=NCHW --input_shape="x:1,3,48,320" --soc_version=Ascend310 --insert_op_conf="rec_aipp.cfg"
```

执行完模型转换脚本后，会生成相应的.om模型文件。更多ATC工具细节请参考[链接](https://www.hiascend.com/document/detail/zh/canncommercial/504/inferapplicationdev/atctool)。

### 3.3 可用模型获取

此处提供了转换好的模型om文件：[resnet50](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/Financial_bills-OCR/resnet50.om)、[db](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/Financial_bills-OCR/db.om  )、[crnn](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/Financial_bills-OCR/crnn.om)。下载后将om文件放置在相应的模型目录下（`models/resnet50`、`models/db`和`models/crnn`）



## 4 编译与运行
**步骤1** 编译后处理插件DBPostProcess

参考[链接](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/mxVision/GeneralTextRecognition/src)编译后处理插件，并将编译好的`libDBPostProcess.so`和`libclipper.so`放在项目目录`./lib`下，添加如下环境变量(其中project_path为项目路径)

```
export LD_LIBRARY_PATH={project_path}/lib/:$LD_LIBRARY_PATH
```

**步骤2** 配置pipeline文件

将pipeline中相关模型路径，配置文件路径及插件路径配置正确

**步骤3** 将要识别的票据图片放到`./inputs`目录下，执行如下命令

```
python3 main.py
```

待执行完毕可在`./outputs`目录下查看结果



## 5 精度测试

### 5.1 resnet50精度

本精度测试在昇腾910芯片上执行，

输入如下命令：

```
python eval.py --data_path=./imagenet2012/val --checkpoint_file_path=./output1/checkpoint/resnet_1-2000_3.ckpt --config_path=./config/resnet50_imagenet2012_config.yaml
```

精度如下：

![](imgs\resnet50精度.jpg)



## 5.2 DB+CRNN端到端精度

测试数据可在[此处](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/Financial_bills-OCR/eval_data.zip)下载，将下载的数据解压到`eval_data`目录下，在项目目录下执行

```
python3 eval.py
```

可得精度结果如下所示

![](imgs\e2e精度.jpg)

精度结果如下表所示

| 模型      | 精度   |
| --------- | ------ |
| resnet50  | 0.9931 |
| 端到端OCR | 0.9531 |



## 6 性能测试

npu性能测试采用ais-infer工具，ais-infer工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[推理工具ais-infer官方源码仓](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。

将三个模型的om文件放到`ais_infer.py`同目录下，执行如下命令可得到三个模型的npu性能

```
python3 ais_infer.py --model model/resnet50.om --batchsize 1 --loop 10
```

```
python3 ais_infer.py --model model/db.om --batchsize 1 --loop 10 --dymHW 1216,1280
```

```
python3 ais_infer.py --model model/crnn.om --batchsize 1 --loop 10
```

gpu性能测试采用使用TensorRT，将三种模型的onnx文件放在装有对应gpu卡（T4）的服务器上，执行如下命令得到gpu性能

```
trtexec --onnx=resnet50.onnx --fp16 --shapes=image:1x3x224x224
```

```
trtexec --onnx=db.onnx --fp16 --shapes=x:1x3x1216x1280
```

```
trtexec --onnx=crnn.onnx --fp16 --shapes=x:1x3x48x320
```

可得性能如下表所示：

| 模型     | gpu性能（T4） | 310性能（4卡） |
| -------- | ------------- | -------------- |
| resnet50 | 903.4893 fps  | 1743.1472 fps  |
| db       | 36.5123 fps   | 45.4676 fps    |
| crnn     | 1159.875 fps  | 1114.392 fps   |



## 7 常见问题

### 7.1 后处理插件路径问题

![](imgs\错误1.png)

**问题描述：**

提示libclipper.so: cannot open shared object file: No such file or directory

**解决方案：**

这是由于没有将后处理插件的路径配置到环境变量中，添加如下环境变量，`project_path`为项目路径

```
export LD_LIBRARY_PATH={project_path}/lib/:$LD_LIBRARY_PATH
```

### 7.2 字体资源问题

![](imgs\错误2.png)

**问题描述：**

提示cannot open resource

**解决方案：**

将字体文件路径正确配置到`eval.py`中`add_text`函数中，如下

```
fontstyle = ImageFont.truetype("SIMSUN.TTC", textSize, encoding="utf-8")
```

