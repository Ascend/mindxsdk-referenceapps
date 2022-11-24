# ArtCoder

## 1. 简介

ArtCoder 是一项二维码图像风格化的项目，主要应用于二维码图像的美化/风格化等场景，本项目基于 MindSpore 框架进行开发，在昇腾平台上进行测试。

## 2. 环境依赖


- 支持的硬件形态和操作系统版本

| 硬件环境                             | 操作系统版本          |
| ------------------------------------ | --------------------- |
| Linux-x86_64 + CPU / GPU / Ascend910 | Ubuntu LTS >= 18.04.1 |
| Linux-aarch64+ CPU / GPU / Ascend310 | Ubuntu LTS >= 18.04.1 |

- 软件依赖

| 软件名称  | 版本     |
| --------- | -------- |
| opencv    | 4.6.0.66 |
| opencv-contrib | 1.23.3 |
| python    | 3.9.2    |
| pillow    | 9.3.0    |
| mindspore | 1.8.1    |


## ３. 目录介绍

本项目名称为 ArtCoder，其工程目录如下：

```bash
.
├── datasets
|   └── download_datasets.sh    // 数据集下载脚本
├── mindspore	            // mindspore 版
│   ├── artcoder.py
│   ├── main.py
│   ├── model
│   │   └── download_model.sh   // 预训练模型下载脚本
│   ├── output	            // 风格化二维码图片输出目录
│   ├── requirements.txt    // Python 库依赖
│   ├── run.sh	            // 运行脚本
│   ├── scan.py	            
│   ├── scan.sh	            // 风格化二维码扫码测试
│   ├── ss_layer.py
│   ├── test.sh	            // 批量测试生成风格化二维码
│   ├── utils.py
│   ├── var_init.py
│   └── vgg.py
├── .gitignore	
├── LICENSE
└── README.md
```

## 4. 准备

正式运行项目前，需要先根据第 **2** 节的要求，配置好相应的运行环境；然后根据提供的脚本下载好数据集和预训练模型。

### 4.1 数据集

下载风格化二维码的开源数据集：

```bash
cd datasets
bash download_datasets.sh
cd ..
```

### 4.2 模型

下载 MindSpore 框架下的开源 VGG-16 预训模型：

```bash
cd mindspore/model/
bash download_model.sh
cd ../../
```


## 5. 运行

首先进入到对应的代码目录中：

```bash
cd mindspore/
```

根据不同的硬件平台修改 `artcoder.py` 文件中对应的硬件指定代码：

```bash
# context.set_context(mode=context.GRAPH_MODE, device_target='CPU')		# CPU 平台
# context.set_context(mode=context.GRAPH_MODE, device_target='GPU')		# GPU 平台
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')	# Ascend 平台
```


然后根据以下注释编缉 `run.sh` 脚本：

```bash
#!/bin/bash

epoch=100                                       # 迭代优化次数
style_img_path="../datasets/style/square.jpg"   # 风格图像路径 
content_img_path="../datasets/content/man.jpg"	# 内容图像路径
code_img_path="../datasets/code/man.jpg"		# 二维码图像路径
output_dir="./output/man/"				        # 输出目录
export OMP_NUM_THREADS=1
python -W ignore main.py --epoch ${epoch} --style_img_path ${style_img_path} --content_img_path ${content_img_path} --code_img_path ${code_img_path} --output_dir ${output_dir}
```

修改好对应的图像路径后，运行 `run.sh` 脚本：

```bash
bash run.sh
```

待程序跑完后可在输出目录下找到最终输出的风格化图像。

## 6. 测试

如果想要批量地进行测试，可以运行 `test.sh` 脚本，根据以下注释修改对应参数：

```bash
mode='some'                         # 测试模式
epoch=100
style_dir="../datasets/style"       # 风格图片目录
content_dir="../datasets/content"   # 内容图片目录
code_dir="../datasets/code"         # 二维码图片目录
output_dir="./output"               # 输出目录
# ...
```

由于风格图像比较多，故提供两种测试模式：`all` 和 `some`，分别指使用所有的风格图进行测试和使用部分风格图进行测试。如果使用 `some`，其所使用的部分风格图列表在 `test.sh` 也可自行修改，即对应下面代码中的 `style_list`，可按需修改成自定义选定的测试图像列表：

```bash
# 部分风格图测试
style_list=('candy.jpg' 'hb.jpg' 'been.jpg' 'st.jpg' 'square.jpg' 'udnie.jpg')    # 选定的测试图像列表
for style in ${style_list[@]}; do
    style_name=$(basename ${style} .jpg)
    style_img_path="${style_dir}/${style_name}.jpg"
    output_img_path="${output_dir}/${image_name}_${style_name}"
    python -W ignore main.py --epoch ${epoch} --style_img_path ${style_img_path} --content_img_path ${content_img_path} --code_img_path ${code_img_path} --output_dir ${output_img_path}
done
```

然后运行 `scan.sh` 脚本，根据以下注释修改对应参数：

```bash
scan_dir="./output"                 # 扫码测试目录
python scan.py --scan_dir ${scan_dir}
```

最后会打印出能正确识别出的风格二维码的准确度。二维码识别使用了 Opencv-Contrib 中带的微信二维码扫码器。


## 7. 效率

> 以下测速均在单进程下进行，涉及 GPU 和 NPU 也都是只使用单张卡进行测速。本测试主要对比对象为 [Pytorch 版的开源 ArtCoder](https://github.com/SwordHolderSH/ArtCoder)。

针对一张 $592 \times 592$ 的二维码图风格化迁移总流程的实际硬件平台测速，包含模型载入、风格化总流程（图像读取、图像 Resize、图像变换、图像风格化迭代优化以及图像保存等步骤），每张二维码图风格化迭代优化 $100$ 个 Epoch。风格化总流程测速包含到图像读取、图像变换等 CPU 操作，受 CPU 型号及服务器上其它 CPU 任务影响较大。为了更好地比较该模型分别基于 PyTorch 的 GPU 平台，和 MindSpore 的 NPU 平台的效率，分别对模型载入、风格化总流程以及风格化流程中的迭代优化进行测速（其中 Mindspore 框架的模型载入时间不稳定，多次测量变化较大）：

| 框架            | 硬件                                                         | 模型载入(s) | 风格化总流程(s) | 迭代优化(s) |
| --------------- | ------------------------------------------------------------ | ----------- | --------------- | ----------- |
| PyTorch 1.8.1   | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz + NVIDIA GeForce RTX 2080 Ti GPU | 6.60        | 11.52           | 11.25       |
| PyTorch 1.8.1   | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz                    | 1.96        | 662.31          | 656.21      |
| PyTorch 1.8.1   | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz + NVIDIA Tesla V100-PCIE GPU | 4.88        | 9.19            | 8.79        |
| MindSpore 1.8.1 | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz + NVIDIA GeForce RTX 2080 Ti GPU | 15.29       | 11.44           | 10.03       |
| MindSpore 1.8.1 | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz                    | 80.78       | 21.15           | 20.76       |
| MindSpore 1.8.1 | Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz + Ascend 910 NPU    | 17.30       | 23.14           | 19.49       |


## 8. 参考

- MindSpore 框架下预训练的 VGG-16 模型: https://www.hiascend.com/zh/software/modelzoo/models/detail/C/0383dcd3e1d444d5ba48fe17da30f0d5/1
- 二维码风格化模型参考：https://github.com/SwordHolderSH/ArtCoder