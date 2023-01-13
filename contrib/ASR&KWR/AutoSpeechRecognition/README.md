# Auto Speech Recognition

## 1 介绍
  本开发样例基于MindX SDK实现了端到端的自动语音识别（Automatic speech recognition, ASR）。<br/>
  ASR主要分为两个步骤：<br/>

  **一、 将语音转换成对应的拼音** <br/>
  **二、 将拼音转换成对应的文字** <br/>

  对于第一步将语音转换为对应的拼音的声学模型我们采用的模型是Google在2020年提出的Conformer模型：[Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)；<br/>
  对于第二步语言模型我们采用的是transformer模型。<br/>
  这两个模型的主要参考代码： [https://github.com/Z-yq/TensorflowASR](https://github.com/Z-yq/TensorflowASR)

### 1.1 支持的产品

支持昇腾310芯片

### 1.2 支持的版本

支持21.0.4版本

版本号查询方法，在Atlas产品环境下，运行命令：

```
npu-smi info
```
### 1.3 特性及适用场景

本项目适用于wav中文语音数据，每条语音3~5秒。

### 1.4 代码目录结构与说明

本sample工程名称为AutoSpeechRecognition，工程目录如下图所示：
```
.
|-------- data
|           |---- lm_tokens.txt                //字典文件
|           |---- sample                       //样例数据集
|           |---- npy                          //样例数据集
|                  |---- feat_data             //语音文件转换的npy文件
|                  |---- len_data              //语音文件转换的npy文件
|-------- model
|           |---- am_conform_batch_one.om      //conformer声学模型
|           |---- lm_transform_batch_one.om    //transformer语言模型
|-------- pipline
|           |---- am_lm.pipeline               //声学模型-语言模型流水线配置文件
|-------- main.py                              //推理及精度测试程序
|-------- post_process.py                      //将推理的结果解码成文字
|-------- pre_process.py                       //对语音数据进行特征提取和对齐
|-------- run.sh                               //样例运行脚本
|-------- README.md
```

## 2 环境依赖

环境依赖软件和版本如下表：

|软件名称    | 版本     |
|-----------|----------|
| python    | 3.9.2    |
| numpy     | 1.23.5   |
| MindX SDK | 3.0.RC3  |
| librosa   | 0.9.2    |
| CANN      | 6.0.RC1  |

1. 请确认环境已安装pip3后，使用pip3 install * 安装以上依赖

2. 请注意MindX SDK使用python版本为3.9.2，如出现无法找到python对应lib库请在root下安装python3.9开发库  
`apt-get install libpython3.9`  
librosa安装若无法编译相关依赖，可参考下述指令在root用户下安装对应的库
```shell
apt-get install llvm-10 -y
LLVM_CONFIG=/usr/lib/llvm-10/bin/llvm-config pip install librosa
apt-get install libsndfile1 -y
apt-get install libasound2-dev libsndfile-dev
apt-get install liblzma-dev
```
3. run.sh脚本中LD_LIBRARY_PATH设置了ACL动态库链接路径为/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64，如果实际环境中路径不一致，需要替换为实际的目录。

4. 如果环境变量中没有MX_SDK_HOME变量，则需要在run.sh脚本中设置MX_SDK_HOME变量为你实际的MX_SDK安装目录。

## 3 模型获取与转换

### 3.1 模型获取

> 由于gitee对于文件大小的限制，om模型超过100M无法上传，可以从以下链接获取并放到项目的 model 目录下：<br/>
>
> [模型下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ASR%26KWR/AutoSpeechRecognition/model.zip)

### 3.2 模型转换
由于原模型是tensorflow的模型，因此我们需要借助于ATC工具将tensorflow的pb模型转化为om模型。
模型转换时应先按照 [准备动作](https://support.huaweicloud.com/atc-model-convert-cann202infer/atlasatc_16_0005.html) 配置好环境和设置环境变量，然后再分别执行以下命令

- 声学模型的转换

`atc --model=./frozen_graph_conform.pb --framework=3 --output=./am_conform_batch_one --input_format=NHWC --input_shape="features:1,1001,80,1;length:1,1" --soc_version=Ascend310 --log=error`

> 声学模型的输入是经过预处理后的数据。除了要进行特征提取外，还要与模型的输入维度对齐。声学模型的输入有两个，第一个是经过预处理后的音频数据，第二个是一个表示语音数据识别出文字长度的一个整形数据。

- 语言模型的转换

`atc --model=./frozen_graph_transform.pb --framework=3 --output=./lm_transform_batch_one --input_format=ND --input_shape="inputs:1,251" --soc_version=Ascend310  --log=error`

> 为了简化推理过程，我们直接把声学模型的输出作为语言模型的输入，所以这里语言模型的输入要与声学模型的输出保持一致



## 4 运行

### 4.1 数据集准备

此模型使用的数据集为[AISHELL-1_sample样例数据集](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ASR%26KWR/AutoSpeechRecognition/data_sample.zip)。下载后将内含的所有wav及txt文件放至"data/sample"目录下。<kbd>data/BAC009S0009W0121.wav</kbd>为其中一条语音，其对应的文字是：其中有两个是内因的指标。

### 4.2 执行以下脚本
```bash
bash run.sh
```
运行样例数据集上的推理及精度、性能测试。

## 5 其它说明

由于模型输入的限制，推理时wav语音的时长应控制在10s及其以下，超过10s的部分会被截断。
