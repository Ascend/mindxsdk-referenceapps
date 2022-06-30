# Mind SDK 风格转换参考设计

## 介绍

本文中风格转换中的地图转换。它通过一种无监督的少样本的学习方式，能够实现卫星地图和卫星地图之间的转换。

论文原文：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

测试集下载地址：https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/maps.zip

初始模型、推理模型及转换脚本下载：https://www.hiascend.com/zh/software/modelzoo/detail/1/3ba3b04fd4964d9b81974381b73f491d

### 支持的产品以及环境依赖

支持 Atlas 200dk开发者套件、Ascend 310推理芯片。

|   软件名称    |    版本     |
| :-----------: | :---------: |
|    ubantu     |  18.04.5 LTS |
|   MindX SDK   |    2.0.4    |
|    Python     |    3.9.2    |
|     CANN      |    5.0.4    |
|     numpy     |   1.22.3    |
| opencv-python |    4.5.5    |

版本号查询方法，在Atlas产品环境下，运行命令：

```
npu-smi info

```
运行后会在终端输出：
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

### 软件方案介绍

| 序号 | 子系统  | 功能描述 |
| 1   | 图像输入 | 调用MindX SDK的appsrc输入图片|
| 2   | 图像解码 | 调用MindX SDK的mxpi_decode对图像解码|
| 2   | 图像放缩 | 调用MindX SDK的mxpi_imageresize，放缩到256*256大小 |
| 3   | 图像推理 | 调用MindX SDK的mxpi_tensorinfer推理图像|
| 7   | 结果输出 | 输出图片信息|

### 代码目录结构与说明

    本参考设计工程名称为Styletransfer，工程目录如下图所示：  
```
StyleTransfer
.
├── README.md
├── models       
│   └── aipp_CycleGAN_pth.config  //aipp配置文件  
├── pipeline
│   └── styletransfer.pipeline
└── src
    └── main.py
```

### python第三方库

```
numpy == 1.16.6

Pillow == 8.2.0

opencv-python == 4.5.2

sympy == 1.4

decorator == 4.4.2

requests == 2.22.0

tqdm == 4.61.0

PyYAML == 5.4.1
```

### 开发准备

> 模型转换

**步骤1** 将pth模型转换为onnx模型
首先在ModelZoo下载CycleGAN模型。

下载地址：https://www.hiascend.com/zh/software/modelzoo/detail/1/3ba3b04fd4964d9b81974381b73f491d

**步骤2** 设置环境变量

```
bash env.sh 

```

**步骤3** 将原始pth模型转化为onnx模型

```
    python3 CycleGAN_onnx_export.py \

--model_ga_path=latest_net_G_A.pth      \

--model_ga_onnx_name=model_Ga.onnx       \
```

**步骤4** 配置AIPP

```
aipp_op{
    aipp_mode:static
    input_format : YUV420SP_U8
    csc_switch : true
    rbuv_swap_switch : true
    src_image_size_w : 256
    src_image_size_h : 256
    min_chn_0 : 127.5
    min_chn_1 : 127.5
    min_chn_2 : 127.5
    var_reci_chn_0: 0.007843137254901
    var_reci_chn_1: 0.007843137254901
    var_reci_chn_2: 0.007843137254901
    matrix_r0c0: 256
    matrix_r0c1: 0
    matrix_r0c2: 359
    matrix_r1c0: 256
    matrix_r1c1: -88
    matrix_r1c2: -183
    matrix_r2c0: 256
    matrix_r2c1: 454
    matrix_r2c2: 0
    input_bias_0: 0
    input_bias_1: 128
    input_bias_2: 128
}
```
保存在/models下的aipp_CycleGAN_pth.config文件中。

**步骤5** 将onnx模型转换为om模型

```
atc --framework=5 --model=model_Ga.onnx --output=sat2map --input_format=NCHW --input_shape="img_sat_maps:1,3,256,256" --out_nodes="Tanh_146:0" --log=debug --soc_version=Ascend310 --insert_op_conf=aipp_CycleGAN_pth.config
```

转换完成后在存放在/models中。

**步骤6** 运行程序

```
python main.py
```

###  运行结果

生成的地图存放在result目录中。
```
ls ../result/map.jpg 
```