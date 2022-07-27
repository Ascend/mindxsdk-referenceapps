# SRflow图像超分辨

## 1 介绍

SRflow图像超分项目是基于MindX SDK 2.0.1 mxVision开发图像超分辨率程序，使用昇腾310芯片进行推理。输入为宽和高均小于等于256的低分辨率图像，经过SRflow模型的推理以及后处理，输出高分辨率图像。

### 1.1 支持的产品

本产品以昇腾310（推理）卡为硬件平台。

### 1.2 支持的版本

本样例配套的CANN版本为 [5.0.4](https://www.hiascend.com/software/cann/commercial) ，MindX SDK版本为 [2.0.4](https://www.hiascend.com/software/Mindx-sdk) 。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 软件方案介绍

本程序采用python开发，前后处理基于python图像处理库PIL。本程序要求输入的图片分辨率小于256*256，格式为jpg或者png。程序首先对输入图片进行padding、归一化、转置等前处理操作，将其转化为适合模型输入的shape为(3，256，256)的向量；然后将该向量转化为mxVision可以处理的元数据形式发送到pipline。利用图像超分辨率模型SRflow获取得到图片超分辨率重建结果，该结果为一shape为(3,2048,2048)的向量。最后，后处理得到2048 \* 2048的高分辨图像。将该图像由padding输入得到的部分切割除去，就可以得到正确的输出。

### 1.4 代码目录结构与说明

```
├── model
│   ├── srflow_df2k_x8_bs1.om # om模型
├── dataset # 验证集
│   └── div2k-validation-modcrop8-gt # 100张高分辨图像
|	└── div2k-validation-modcrop8-8x # 100张对应的8倍降采样低分辨图像   
├── images # main.py处理的，作为例子的图片
│   ├──	0802.png # 低分辨率图片
|   └── 0802_hr.png #对应的高分辨率原图像
├── result # 存放输出图像的文件夹
├── main.py # 主程序
├── utils.py # 工具包
├── evaluate.py # 精度验证程序
```



## 2 环境依赖 

### 2.1 软件版本

| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 2.0.4        | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| Ascend-CANN-toolkit | 5.0.4        | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | Ubuntu 18.04 | 操作系统                      | Ubuntu官网获取                                               |

### 2.2 准备工作

#### 2.21 环境变量

在项目运行前，需要设置环境变量：

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
. ${SDK安装路径}/mxVision/set_env.sh

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

注：其中SDK安装路径${MX_SDK_HOME}替换为用户的SDK安装路径。

#### 2.22 模型下载

该项目依赖于modelzoo中的模型srflow，可以直接下载[模型软件包](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/46c524de25e040239fc2e4a7e15113b4)，将软件包目录下srflow_df2k_x8_bs1.om模型放入本项目，model地址下。

#### 2.23 修改脚本

在`main.py`和`evaluate.py`中配置 `srflow_df2k_x8_bs1.om` 模型路径

```python
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "appsrc0",
                    "modelPath": "./model/srflow_df2k_x8_bs1.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "appsink0"
            },
```



## 3 运行

使用本目录下image文件夹中的图片对作为输入。

在本项目目录下，运行脚本：

``` bash
python3 main.py
```

得到的结果为result.jpg , 在终端可以看到生成图像的PSNR值：

```
PSNR: 29.12
Infer finished.
```



## 4 精度验证

PSNR（峰值信噪比）经常用作图像压缩等领域中信号重建质量的测量方法。

### 4.1 准备验证数据集

```
wget  http://data.vision.ee.ethz.ch/alugmayr/SRFlow/datasets.zip
unzip datasets.zip
rm datasets.zip
```

将解压后的datasets文件目录下的div2k-validation-modcrop8-gt 文件夹和 div2k-validation-modcrop8-8x 文件夹放入本项目目录下的dataset文件夹中。

### 4.2 运行精度验证脚本

```
python3 evaluate.py
```

生成的结果位于result文件夹中，生成图像的文件名与低分辨图像的文件名相同。

### 4.3 精度验证结果

每处理一张图片，终端都会输出推理结果的PSNR值：

```
Processing ./dataset/div2k-validation-modcrop8-x8/0825.png...
I20220727 18:04:08.653025 17725 MxpiBufferManager.cpp:121] create host buffer and copy data, Memory size(0).
PSNR: 20.62
Infer finished.
```

在处理完100张图片的验证集后，可以获得平均PSNR值：
![](https://gitee.com/distancemay/mindxsdk-referenceapps/blob/master/contrib/SRflow/image/pnsr.png)

与使用源模型软件包benchmark 推理结果对比结果如下：

```
图像编号  benchmark结果	        本项目实际结果	
0801.png PSNR: 21.48			PSNR: 21.48	
0802.png PSNR: 29.12			PSNR: 29.12	
0803.png PSNR: 31.82			PSNR: 31.82	
0804.png PSNR: 21.59			PSNR: 21.59	
0805.png PSNR: 21.83			PSNR: 21.83	
0806.png PSNR: 23.05			PSNR: 23.05	
0807.png PSNR: 16.09			PSNR: 16.09	
0808.png PSNR: 22.01			PSNR: 22.01	
0809.png PSNR: 23.44			PSNR: 23.44	
0810.png PSNR: 21.76			PSNR: 21.76	
0811.png PSNR: 21.51			PSNR: 21.51	
0812.png PSNR: 22.40			PSNR: 22.40	
0813.png PSNR: 24.99			PSNR: 24.99	
0814.png PSNR: 23.53			PSNR: 23.53	
0815.png PSNR: 28.07			PSNR: 28.07	
0816.png PSNR: 25.44			PSNR: 25.44	
0817.png PSNR: 26.13			PSNR: 26.13	
0818.png PSNR: 23.04			PSNR: 23.04	
0819.png PSNR: 21.41			PSNR: 21.41	
0820.png PSNR: 18.15			PSNR: 18.15	
0821.png PSNR: 24.37			PSNR: 24.37	
0822.png PSNR: 22.93			PSNR: 22.93	
0823.png PSNR: 18.89			PSNR: 18.89	
0824.png PSNR: 23.22			PSNR: 23.22	
0825.png PSNR: 20.62			PSNR: 20.62	
0826.png PSNR: 19.73			PSNR: 19.73	
0827.png PSNR: 27.42			PSNR: 27.42	
0828.png PSNR: 15.77			PSNR: 15.77	
0829.png PSNR: 23.80			PSNR: 23.80	
0830.png PSNR: 20.39			PSNR: 20.39	
0831.png PSNR: 23.95			PSNR: 23.95	
0832.png PSNR: 22.10			PSNR: 22.10	
0833.png PSNR: 26.93			PSNR: 26.93	
0834.png PSNR: 20.73			PSNR: 20.73	
0835.png PSNR: 18.22			PSNR: 18.22	
0836.png PSNR: 19.88			PSNR: 19.88	
0837.png PSNR: 20.29			PSNR: 20.29	
0838.png PSNR: 29.84			PSNR: 29.84	
0839.png PSNR: 22.64			PSNR: 22.64	
0840.png PSNR: 22.34			PSNR: 22.34	
0841.png PSNR: 22.70			PSNR: 22.70	
0842.png PSNR: 23.87			PSNR: 23.87	
0843.png PSNR: 35.56			PSNR: 35.56	
0844.png PSNR: 31.73			PSNR: 31.73	
0845.png PSNR: 18.22			PSNR: 18.22	
0846.png PSNR: 19.49			PSNR: 19.49	
0847.png PSNR: 22.00			PSNR: 22.00	
0848.png PSNR: 23.11			PSNR: 23.11	
0849.png PSNR: 16.05			PSNR: 16.05	
0850.png PSNR: 24.15			PSNR: 24.15	
0851.png PSNR: 23.65			PSNR: 23.65	
0852.png PSNR: 24.89			PSNR: 24.89	
0853.png PSNR: 26.35			PSNR: 26.35	
0854.png PSNR: 18.80			PSNR: 18.80	
0855.png PSNR: 22.59			PSNR: 22.59	
0856.png PSNR: 18.89			PSNR: 18.89	
0857.png PSNR: 32.11			PSNR: 32.11	
0858.png PSNR: 22.49			PSNR: 22.49	
0859.png PSNR: 19.27			PSNR: 19.27	
0860.png PSNR: 16.46			PSNR: 16.46	
0861.png PSNR: 17.96			PSNR: 17.96	
0862.png PSNR: 28.54			PSNR: 28.54	
0863.png PSNR: 28.62			PSNR: 28.62	
0864.png PSNR: 22.79			PSNR: 22.79	
0865.png PSNR: 23.03			PSNR: 23.03	
0866.png PSNR: 20.03			PSNR: 20.03	
0867.png PSNR: 21.88			PSNR: 21.88	
0868.png PSNR: 27.02			PSNR: 27.02	
0869.png PSNR: 19.66			PSNR: 19.66	
0870.png PSNR: 21.75			PSNR: 21.75	
0871.png PSNR: 25.76			PSNR: 25.76	
0872.png PSNR: 19.69			PSNR: 19.69	
0873.png PSNR: 19.07			PSNR: 19.07	
0874.png PSNR: 25.00			PSNR: 25.00	
0875.png PSNR: 18.87			PSNR: 18.87	
0876.png PSNR: 18.37			PSNR: 18.37	
0877.png PSNR: 33.31			PSNR: 33.31	
0878.png PSNR: 24.87			PSNR: 24.87	
0879.png PSNR: 21.78			PSNR: 21.78	
0880.png PSNR: 23.19			PSNR: 23.19	
0881.png PSNR: 20.89			PSNR: 20.89	
0882.png PSNR: 29.10			PSNR: 29.10	
0883.png PSNR: 19.39			PSNR: 19.39	
0884.png PSNR: 21.24			PSNR: 21.24	
0885.png PSNR: 18.86			PSNR: 18.86	
0886.png PSNR: 29.92			PSNR: 29.92	
0887.png PSNR: 19.60			PSNR: 19.60	
0888.png PSNR: 24.55			PSNR: 24.55	
0889.png PSNR: 22.18			PSNR: 22.18	
0890.png PSNR: 21.37			PSNR: 21.37	
0891.png PSNR: 21.63			PSNR: 21.63	
0892.png PSNR: 23.06			PSNR: 23.06	
0893.png PSNR: 28.91			PSNR: 28.91	
0894.png PSNR: 24.83			PSNR: 24.83	
0895.png PSNR: 15.69			PSNR: 15.69	
0896.png PSNR: 31.27			PSNR: 31.27	
0897.png PSNR: 13.12			PSNR: 13.12	
0898.png PSNR: 27.51			PSNR: 27.51	
0899.png PSNR: 25.42			PSNR: 25.42	
0900.png PSNR: 21.17			PSNR: 21.17	
Average PSNR: 22.941				
```
可以看到100对值完全相同，精度无任何损失。