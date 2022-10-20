# 基于深度学习的图像配准

## 1 介绍
基于深度学习的图像配准基于 MindXSDK 开发，在晟腾芯片上进行图像配准。输入两幅图片，可以匹配两幅图像中的特征点。

### 1.1 支持的产品

本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.2 支持的版本

本样例配套的CANN版本为[5.0.4](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2Fcann%2Fcommercial)，MindX SDK版本为[2.0.4](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2FMindx-sdk)。

MindX SDK安装前准备可参考[《用户指南》](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)。

### 1.3 软件方案介绍

基于MindX SDK的基于深度学习的图像配准的业务流程为：将输入的两幅图片进行归一化等预处理操作后，输入到模型中进行推理，对输出的关键点，进行极大值抑制去除相近的关键点，再进一步去除靠近边界的关键点，最后利用knn聚类算法得到可能性最大的关键点。本系统的各模块及功能描述如表1.1所示：

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统   | 功能描述                   |
| ---- | -------- | -------------------------- |
| 1    | 图像输入 | 读取图像                   |
| 2    | 预处理   | 对图像进行预处理           |
| 3    | 模型推理 | 对输入进行推理并输出结果   |
| 5    | 后处理   | 从模型推理结果中解出关键点 |

### 1.4 代码目录结构与说明

本工程名称为基于深度学习的图像配准，工程目录如下图所示：

```shell
.
│  README.mdn
│  pth2onnx.py
│  onnx2om.sh
└─python
    │  main.py
    │  requirements.txt
    │  
    ├─config
    │      test.yaml
```



### 1.5 技术实现流程图

<center>
    <img src="./images/pipeline.png">
    <br>
</center>

## 2 环境依赖

环境依赖软件和版本如下表：

| 软件                | 版本                                                         | 说明                          |
| ------------------- | ------------------------------------------------------------ | ----------------------------- |
| mxVision            | [mxVision 2.0.4](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2FMindx-sdk) | mxVision软件包                |
| Ascend-CANN-toolkit | [CANN 5.0.4](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2Fcann%2Fcommercial) | Ascend-cann-toolkit开发套件包 |
| 操作系统            | [Ubuntu 18.04](https://gitee.com/link?target=https%3A%2F%2Fubuntu.com%2F) | Linux操作系统                 |
| OpenCV              | 4.6.0                                                        | 用于结果可视化                |

在编译运行项目前，需要设置环境变量：

在进行模型转换和编译运行前，需设置如下的环境变量：

```shell
export MX_SDK_HOME=${MX_SDK_HOME}
export install_path=${install_path}
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH:.
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:${MX_SDK_HOME}/python:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

注：**${MX_SDK_HOME}** 替换为用户自己的MindX_SDK安装路径（例如："/home/xxx/MindX_SDK/mxVision"）；

 **${install_path}** 替换为开发套件包所在路径（例如：/usr/local/Ascend/ascend-toolkit/latest）。

- 环境变量介绍

```
MX_SDK_HOME：MindX SDK mxVision的根安装路径，用于包含MindX SDK提供的所有库和头文件。
LD_LIBRARY_PATH：提供了MindX SDK已开发的插件和相关的库信息。
install_path：ascend-toolkit的安装路径。
PATH：添加python的执行路径和atc转换工具的执行路径。
LD_LIBRARY_PATH：添加ascend-toolkit和MindX SDK提供的库目录路径。
ASCEND_OPP_PATH：atc转换工具需要的目录。 
```

###  3. 模型转换

模型转换使用的是ATC工具，具体使用教程可参考[《ATC工具使用指南》](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fdoc%2FEDOC1100191944%2Fa3cf4cee)。

###  3.1 车牌检测模型的转换

**步骤1** **模型获取** 将[基于深度学习的图像配准原工程](https://drive.google.com/drive/folders/1h-MH3wEiN7BoLyMRjF1OAwABKqq6gVFL?usp=sharing)下载到**本地**。

**步骤2** **pth转onnx** 将**pth2onnx.py**脚本放至**本地**工程目录下，执行如下命令：

```
python pth2onnx.py
```

按照实际情况修改路径：

```python
if __name__ == '__main__':
    checkpoint = './SuperRetina.pth'
    onnx_path = './SuperRetina.onnx'
    input = torch.randn(2, 1, 768, 768)
    pth_to_onnx(input, checkpoint, onnx_path)
```

*版本要求：*

*Python = 3.8.3*

*Pytorch = 1.7.0*

注：若原工程链接失效，可以直接下载已经转换好的[superRetina.onnx](https://gitee.com/link?target=https%3A%2F%2Fmindx.sdk.obs.cn-north-4.myhuaweicloud.com%2Fmindxsdk-referenceapps%20%2Fcontrib%2FMMNET%2Fmodel.zip)模型。

**步骤3** **onnx转om** 将步骤2中转换获得的onnx模型存放至**服务器端**的CarPlateRecognition/model/目录下，执行如下命令：

```
atc --model=./superRetina.onnx --output=./superRetina --framework=5 --soc_version=Ascend310 --output_type=FP32
```

## 编译与运行

（描述项目安装运行的全部步骤，，如果不涉及个人路径，请直接列出具体执行命令）

示例步骤如下：
**步骤1**  从 https://projects.ics.forth.gr/cvrl/fire/FIRE.7z下载数据集，解压后./FIRE文件夹，放到./data文件夹。

**步骤2**  按照第 2 小节 环境依赖 中的步骤设置环境变量。

**步骤3**  按照第 3 小节 模型转换 中的步骤获得 om 模型文件。

**步骤4**  在./python目录下运行main.py函数，执行如下命令：

```bash
python main.py 
```

结果输出到终端，结果如下所示：

```bash
100%|██████████| 133/133 [1:37:18<00:00, 43.90s/it]
----------------------------------------
Failed:0.00%, Inaccurate:3.01%, Acceptable:96.99%
----------------------------------------
S: 0.949, P: 0.544, A: 0.780, mAUC: 0.758
```