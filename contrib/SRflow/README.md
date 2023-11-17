# SRflow图像超分辨

## 1 介绍

SRflow图像超分项目是基于MindX SDK 2.0.4  mxVision开发图像超分辨率程序，使用昇腾310芯片进行推理。输入为宽和高均小于等于256的低分辨率图像，经过SRflow模型的推理以及后处理，输出高分辨率图像。

### 1.1 支持的产品

本产品以昇腾310（推理）卡为硬件平台。

### 1.2 支持的版本

本样例配套的CANN版本为 [7.0.RC1](https://www.hiascend.com/software/cann/commercial) ，MindX SDK版本为 [5.0.RC3](https://www.hiascend.com/software/Mindx-sdk) 。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 软件方案介绍

本程序采用python开发，前后处理基于python图像处理库opencv。本程序要求输入的图片分辨率小于256*256，格式为jpg或者png。程序首先对输入图片进行padding、归一化、转置等前处理操作，将其转化为适合模型输入的shape为(3，256，256)的向量；然后将该向量转化为mxVision可以处理的元数据形式发送到pipeline。利用图像超分辨率模型SRflow获取得到图片超分辨率重建结果，该结果为一shape为(3,2048,2048)的向量。最后，后处理得到2048 \* 2048的高分辨图像。将该图像由padding输入得到的部分切割除去，就可以得到正确的输出。

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

### 1.5 适用场景
待处理低分辨图像分辨率不小于100*100,长宽均不大于256,图像质量较高,无较大噪声。



## 2 环境依赖 

### 2.1 软件版本

| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 5.0.RC3      | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| Ascend-CANN-toolkit | 7.0.RC1      | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | Ubuntu 18.04 | 操作系统                      | Ubuntu官网获取                                               |

### 2.2 准备工作

#### 2.2.1 环境变量

在项目运行前，需要设置环境变量：

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
. ${SDK安装路径}/mxVision/set_env.sh
```

注：其中SDK安装路径${MX_SDK_HOME}替换为用户的SDK安装路径。

#### 2.2.2 模型下载

该项目依赖于modelzoo中的模型srflow，可以直接下载[模型软件包](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/SRFlow/ATC%20SRFlow%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip)，将软件包目录下srflow_df2k_x8_bs1.om模型放入本项目，model地址下。

#### 2.2.3 修改脚本

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
### 3.1 关于输出图像以及精度验证的说明
&emsp;&emsp;作为图像超分项目，一般要验证该程序的精度与性能，步骤如下：首先找到一张分辨率较高的图片，即hr图片，然后将其进行八倍或者四倍降采样，得到低分辨的、待处理的图片。将待处理的图片padding为256\*256，然后输入模型，得到2048\*2048的超分辨图像，再根据hr图片的大小进行裁剪，获得一张和hr图片大小相同的图片，最后测试PSNR值，获得图像质量评估。大部分验证集和本项目的测试便是采用了这种方法得到高分辨和低分辨的图片对。
&emsp;&emsp;然而，在实际应用中，一般是先有低分辨的图像，想要通过模型提高图像的分辨率。一开始手头不可能有现成的hr图像。

假设现在有一张160\*120的图像:
![在这里插入图片描述](https://img-blog.csdnimg.cn/5dbeedff97804a119859d592f58a0964.png)
在前处理时对其进行padding,使之符合模型256\*256的标准输入:
![请添加图片描述](https://img-blog.csdnimg.cn/df9b6ac36da0406b809ae1d85227be33.png)

输入模型进行推理，得到模型标准输出，即为大小为2048\*2048的超分辨图像:
![请添加图片描述](https://img-blog.csdnimg.cn/7c244b5b384840559202d85b60702229.png)
此时想要通过切割除去padding对应的超分辨部分，需要手动计算hr图像的大小。从256到2048，模型将图像大小放大8倍。而原图像为160\*120，则hr图像长应该为160\*8=1280，宽应为120\*8=960。经过人工处理得到大小为1280\*960的图像作为hr图像，切割可得预期图像:
![请添加图片描述](https://img-blog.csdnimg.cn/0b31f14353664a4295eeda3536ac5677.jpeg)



### 3.2 运行方法

选取一对八倍下采样前后的png格式图片作为样例(可直接使用div2k验证集中的图片对)，将下采样后的图片和hr图片(低分辨图像对应的高分辨图像,即下采样前的图像)放入image文件夹中,将main.py中两个常数IMAGE和HRIMAGE分别改为低分辨和高分辨图像的路径。

在本项目目录下，运行脚本：

```bash
python3 main.py
```

得到的结果为result.jpg , 在终端可以看到生成图像的PSNR值(这里使用div2k验证集中的0802.png作为样例)：

```
PSNR: 29.04
Infer finished.
```



## 4 精度验证

PSNR（峰值信噪比）经常用作图像压缩等领域中信号重建质量的测量方法。

### 4.1 准备验证数据集

在[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/SRFlow/dataset.zip)处下载数据集dataset.zip,放在项目目录下。
```
unzip dataset.zip
rm dataset.zip
```

将解压后的dataset文件目录下的div2k-validation-modcrop8-gt 文件夹和 div2k-validation-modcrop8-8x 文件夹放入本项目目录下的dataset文件夹中。

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


在处理完100张图片的验证集后，可以获得平均PSNR值：22.93 


与使用源模型软件包benchmark 推理结果对比结果如下：

```
原benchmark推理结果			本项目实际结果
0801.png PSNR: 21.48			21.23 
0802.png PSNR: 29.12			29.04 
0803.png PSNR: 31.82			31.76 
0804.png PSNR: 21.59			21.73 
0805.png PSNR: 21.83			21.62 
0806.png PSNR: 23.05			23.56 
0807.png PSNR: 16.09			16.36 
0808.png PSNR: 22.01			22.56 
0809.png PSNR: 23.44			23.21 
0810.png PSNR: 21.76			21.17 
0811.png PSNR: 21.51			22.19 
0812.png PSNR: 22.40			22.17 
0813.png PSNR: 24.99			24.88 
0814.png PSNR: 23.53			23.59 
0815.png PSNR: 28.07			27.74 
0816.png PSNR: 25.44			25.57 
0817.png PSNR: 26.13			25.90 
0818.png PSNR: 23.04			22.75 
0819.png PSNR: 21.41			21.14 
0820.png PSNR: 18.15			17.85 
0821.png PSNR: 24.37			23.83 
0822.png PSNR: 22.93			23.17 
0823.png PSNR: 18.89			20.64 
0824.png PSNR: 23.22			23.09 
0825.png PSNR: 20.62			20.08 
0826.png PSNR: 19.73			19.51 
0827.png PSNR: 27.42			27.66 
0828.png PSNR: 15.77			16.08 
0829.png PSNR: 23.80			23.65 
0830.png PSNR: 20.39			19.89 
0831.png PSNR: 23.95			24.16 
0832.png PSNR: 22.10			22.16 
0833.png PSNR: 26.93			26.71 
0834.png PSNR: 20.73			20.78 
0835.png PSNR: 18.22			18.30 
0836.png PSNR: 19.88			20.25 
0837.png PSNR: 20.29			20.27 
0838.png PSNR: 29.84			21.02 
0839.png PSNR: 22.64			22.56 
0840.png PSNR: 22.34			22.72 
0841.png PSNR: 22.70			22.87 
0842.png PSNR: 23.87			23.77 
0843.png PSNR: 35.56			35.43 
0844.png PSNR: 31.73			35.97 
0845.png PSNR: 18.22			18.55 
0846.png PSNR: 19.49			19.44 
0847.png PSNR: 22.00			21.91 
0848.png PSNR: 23.11			23.32 
0849.png PSNR: 16.05			16.35 
0850.png PSNR: 24.15			24.01 
0851.png PSNR: 23.65			24.08 
0852.png PSNR: 24.89			25.28 
0853.png PSNR: 26.35			26.21 
0854.png PSNR: 18.80			19.71 
0855.png PSNR: 22.59			22.91 
0856.png PSNR: 18.89			17.97 
0857.png PSNR: 32.11			32.20 
0858.png PSNR: 22.49			23.19 
0859.png PSNR: 19.27			19.52 
0860.png PSNR: 16.46			15.97 
0861.png PSNR: 17.96			18.06 
0862.png PSNR: 28.54			28.87 
0863.png PSNR: 28.62			28.48 
0864.png PSNR: 22.79			23.87 
0865.png PSNR: 23.03			23.09 
0866.png PSNR: 20.03			20.34 
0867.png PSNR: 21.88			21.98 
0868.png PSNR: 27.02			26.77 
0869.png PSNR: 19.66			19.72 
0870.png PSNR: 21.75			21.09 
0871.png PSNR: 25.76			25.30 
0872.png PSNR: 19.69			19.27 
0873.png PSNR: 19.07			19.04 
0874.png PSNR: 25.00			24.27 
0875.png PSNR: 18.87			19.22 
0876.png PSNR: 18.37			18.21 
0877.png PSNR: 33.31			33.19 
0878.png PSNR: 24.87			24.77 
0879.png PSNR: 21.78			21.83 
0880.png PSNR: 23.19			23.67 
0881.png PSNR: 20.89			20.91 
0882.png PSNR: 29.10			28.73 
0883.png PSNR: 19.39			19.57 
0884.png PSNR: 21.24			21.08 
0885.png PSNR: 18.86			18.14 
0886.png PSNR: 29.92			29.67 
0887.png PSNR: 19.60			20.25 
0888.png PSNR: 24.55			24.60 
0889.png PSNR: 22.18			21.89 
0890.png PSNR: 21.37			21.58 
0891.png PSNR: 21.63			21.56 
0892.png PSNR: 23.06			22.69 
0893.png PSNR: 28.91			28.97 
0894.png PSNR: 24.83			25.62 
0895.png PSNR: 15.69			14.99 
0896.png PSNR: 31.27			31.38 
0897.png PSNR: 13.12			15.06 
0898.png PSNR: 27.51			27.22 
0899.png PSNR: 25.42			25.45 
0900.png PSNR: 21.17			21.18 
平均PSNR	   22.94		    22.93 
可见精度几乎没有损失，符合精度要求。