# Mind SDK 风格迁移

## 1 介绍

本文中风格转换中的地图转换。它通过一种无监督的少样本的学习方式，能够实现航拍地图和卫星地图之间的相互转换。
论文原文：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

初始模型及转换脚本下载：https://www.hiascend.com/zh/software/modelzoo/detail/1/3ba3b04fd4964d9b81974381b73f491d

### 1.1 支持的产品

支持 Atlas 200dk开发者套件、Ascend 310推理芯片。

### 1.2 支持的版本

支持 Atlas 200dk开发者套件、Ascend 310推理芯片。

版本号查询方法，在Atlas产品环境下，运行命令：

```
npu-smi info

```

```
+------------------------------------------------------------------------------+
| npu-smi 21.0.3.1                     Version: 21.0.3.1                       |
+-------------------+-----------------+----------------------------------------+
| NPU     Name      | Health          | Power(W)          Temp(C)              |
| Chip    Device    | Bus-Id          | AICore(%)         Memory-Usage(MB)     |
+===================+=================+========================================+
| 0       310       | OK              | 12.8              58                   |
| 0       0         | NA              | 0                 5324 / 8192          |
+===================+=================+========================================+
```

### 1.4 代码目录结构与说明

    本sample工程名称为Styletransfer，工程目录如下图所示：
    
```
StyleTransfer
.
├── README.md
├── env.sh                        //环境变量脚本
├── models       
│   └── aipp_CycleGAN_pth.config  //atc 配置文件  
├── pipeline
│   └── styletransfer.pipeline
└── src
    └── main.py
```


### 2 环境依赖

推荐系统为：Linux davinci-mini arch64 GNU/Linux




### 3 代码实现

示例步骤如下：

### 3.1 模型转换

**步骤1** 将pth模型转换为onnx模型

设置环境变量
```
bash env.sh 

```
将原始pth模型转化为onnx模型
```

    python3 CycleGAN_onnx_export.py \

--model_ga_path=latest_net_G_A.pth      \

--model_ga_onnx_name=model_Ga.onnx       \

```


**步骤2** 将onnx模型转换为om模型

```
atc --framework=5 --model=model_Ga.onnx --output=sat2map --input_format=NCHW --input_shape="img_sat_maps:1,3,256,256" --out_nodes="Tanh_146:0" --log=debug --soc_version=Ascend310 --insert_op_conf=aipp_CycleGAN_pth.config

```


**步骤3** 运行程序

```
cd src/

python main.py

```

### 3.2 运行结果

生成的地图存放在result目录中。

```
ls ../result/map.jpg 

```


### 4 软件依赖说明

### 4.1 软件依赖


```
CANN 5.0.2.alpha003

torch == 1.5.0

torchvision == 0.9.0

onnx==1.7.0

onnx-simplifier==0.3.6

onnxconverter-common==1.6.1

onnxoptimizer==0.2.6

onnxruntime==1.6.0

tensorboard==1.15.0

tensorflow==1.15.0

tensorflow-estimator ==1.15.1

termcolor==1.1.0

```

### 4.2 python第三方库

```

numpy == 1.16.6

Pillow == 8.2.0

opencv-python == 4.5.2.52

sympy == 1.4

decorator == 4.4.2

requests == 2.22.0

tqdm == 4.61.0

PyYAML == 5.4.1
```
