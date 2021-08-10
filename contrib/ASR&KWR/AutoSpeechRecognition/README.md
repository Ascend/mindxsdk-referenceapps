# Auto Speech Recognition

## 1 简介
  本开发样例基于MindX SDK实现了端到端的自动语音识别（Automatic speech recognition, ASR）。<br/>
  ASR主要分为两个步骤：<br/>

  **一、 将语音转换成对应的拼音** <br/>
  **二、 将拼音转换成对应的文字** <br/>

  对于第一步将语音转换为对应的拼音的声学模型我们采用的模型是Google在2020年提出的Conformer模型：[Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)；<br/>
  对于第二步语言模型我们采用的是transformer模型。<br/>
  这两个模型的主要参考代码： [https://github.com/Z-yq/TensorflowASR](https://github.com/Z-yq/TensorflowASR)


## 2 模型转换
由于原模型是tensorflow的模型，因此我们需要借助于ATC工具将tensorflow的pb模型转化为om模型。
模型转换时应先按照 [准备动作](https://support.huaweicloud.com/atc-model-convert-cann202infer/atlasatc_16_0005.html) 配置好环境和设置环境变量，然后再分别执行以下命令


- 声学模型的转换

`atc --model=./frozen_graph_conform.pb --framework=3 --output=./am_conform_batch_one --input_format=NHWC --input_shape="features:1,1001,80,1;length:1,1" --soc_version=Ascend310 --log=error`

> 声学模型的输入是经过预处理后的数据。除了要进行特征提取外，还要与模型的输入维度对齐。声学模型的输入有两个，第一个是经过预处理后的音频数据，第二个是一个表示语音数据识别出文字长度的一个整形数据。

- 语言模型的转换

`atc --model=./frozen_graph_transform.pb --framework=3 --output=./lm_transform_batch_one --input_format=ND --input_shape="inputs:1,251" --soc_version=Ascend310  --log=error`

> 为了简化推理过程，我们直接把声学模型的输出作为语言模型的输入，所以这里语言模型的输入要与声学模型的输出保持一致

## 3 目录结构


```
.
|-------- data
|--------   |---- BAC009S0009W0133.wav         //样例原始数据
|--------   |---- feat_data_sample.npy         //样例数据
|--------   |---- len_data_sample.npy          //样例数据
|--------   |---- lm_tokens.txt                //字典文件
|-------- model
|           |---- am_conform_batch_one.om      //conformer声学模型
|           |---- lm_transform_batch_one.om    //transformer语言模型
|-------- pipline
|           |---- am_lm.pipeline               //声学模型-语言模型流水线配置文件
|-------- main.py                              //测试样例
|-------- post_process.py                      //将推理的结果解码成文字
|-------- pre_process.py                       //对语音数据进行特征提取和对齐
|-------- run.sh                               //样例运行脚本
|-------- README.md
```

## 4 依赖

|软件名称    | 版本     |
|-----------|----------|
| python    | 3.7.5    |
| numpy     | 1.18.2   |
| MindX SDK | 2.0.2    |
| librosa   | 0.8.0    |
请确认环境已安装pip3后，使用pip3 install * 安装以上依赖

请注意MindX SDK使用python版本为3.7.5，如出现无法找到python对应lib库请在root下安装python3.7开发库  
`apt-get install libpython3.7`  
librosa安装若无法编译相关依赖，可参考下述指令在root用户下安装对应的库
```shell
apt-get install llvm-10 -y
LLVM_CONFIG=/usr/lib/llvm-10/bin/llvm-config pip install librosa
apt-get install libsndfile1 -y
apt-get install libasound2-dev libsndfile-dev
apt-get install liblzma-dev
```

## 5 运行

1. 获取om模型
> 由于gitee对于文件大小的限制，om模型超过100M无法上传，可以从以下链接获取并放到项目的 model 目录下：<br/>
> [om模型下载链接](https://pan.baidu.com/s/1AvJ1m_olBqvIOKoxKbt3aA) 提取码：4s4e
> [pb模型下载链接](https://pan.baidu.com/s/1lUquThnwRv7zGcyIz2x_ww) 提取码：0wam
2. run.sh脚本中LD_LIBRARY_PATH设置了ACL动态库链接路径为/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64，如果实际环境中路径不一致，需要替换为实际的目录。
3. 如果环境变量中没有MX_SDK_HOME变量，则需要在run.sh脚本中设置MX_SDK_HOME变量为你实际的MX_SDK安装目录。
4. 执行以下脚本
```bash
bash run.sh
```

## 6 其它说明

1. 此模型使用的数据集为[AISHELL数据集](http://www.aishelltech.com/kysjcp)。<kbd>data/BAC009S0009W0133.wav</kbd>为其中一条语音，其对应的文字是：宝龙地产的收入较二零一三年增加约百分之三十三点一。
2. 由于模型输入的限制，推理时wav语音的时长应控制在10s及其以下，超过10s的部分会被截断。
