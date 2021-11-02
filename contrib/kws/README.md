# Keyword Spotting KWS

## 1 简介
  本开发样例基于MindX SDK实现了端到端的关键词检测（Keyword Spotting KWS）。<br/>
  所选择的关键词为：上海 北京 政策 中国 记者 城市 <br/>
  kws主要分为两个步骤：<br/>

  **一、 构建声学模型** <br/>
  **二、 对模型输出进行解码，查看是否出现目标关键词** <br/>

  声学模型采用CRNN-CTC,模型构建参考论文《CRNN-CTC Based Mandarin Keyword Spotting》<br/>

## 2 模型转换
由于原模型是onnx模型（[下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/kws/model.zip)），需借助ACT工具将onnx模型转换为om模型。模型转换应先按照[准备动作](https://support.huaweicloud.com/atc-model-convert-cann202infer/atlasatc_16_0005.html)  
配置好环境和设置环境变量，然后执行以下命令

```bash
atc --framework=5 --model={model_path} --input_shape="input:1,80,1464"
    --output=am_batch_one  --soc_version=Ascend310
```


## 3 目录结构

```
.
|-------- data
|--------   |---- BAC009S0102W0436.wav         //样例原始数据
|--------   |---- data.yaml                    //参数
|--------   |---- mean_std.npz                 //特征的均值和标准差 用于特征标准化
|-------- model
|           |---- crnn_ctc
|                      |----am_batch_one.om    //crnn声学模型
|-------- pipeline
|           |---- crnn_ctc.pipeline            //声学模型流水线配置文件
|-------- python
|           |---- main.py                      //测试样例
|           |---- performance_test.py          //模型性能测试
|           |---- post_process.py              //将推理的结果解码成文字
|           |---- preprocessing.py             //对语音数据进行特征提取
|           |---- run.sh                       //样例运行脚本
|-------- README.md
```

## 4 依赖

|软件名称    | 版本     |
|-----------|----------|
| python    | 3.7.5    |
| MindX SDK | 2.0.2    |
| librosa   | 0.8.1    |
| pypinyin  | 0.42.0   |
| torch     | 1.9.0    |
| PyYAML    | 5.4.1    |
| overrides | 6.1.0    |
请确认环境已安装pip后，使用pip install * 安装以上依赖<br/>
如果环境中有多个版本的python,请确认环境已安装pip3后，使用pip3 install * 安装以上依赖<br/>
请注意MindX SDK使用python版本为3.7.5，如出现无法找到python对应lib库请在root下安装python3.7开发库
```bash
apt-get install libpython3.7
```
librosa安装若无法编译相关依赖，可参考下述指令在root用户下安装对应的库<br/>
```bash
apt-get install llvm-10 -y
LLVM_CONFIG=/usr/lib/llvm-10/bin/llvm-config pip install librosa
apt-get install libsndfile1 -y
apt-get install libasound2-dev libsndfile-dev
apt-get install liblzma-dev
```


## 5 运行

1. 获取om模型
2. run.sh脚本中LD_LIBRARY_PATH设置了ACL动态库链接路径为/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64，如果实际环境中路径不一致，需要替换为实际的目录。
3. 如果环境变量中没有MX_SDK_HOME变量，则需要在run.sh脚本中设置MX_SDK_HOME变量为你实际的MX_SDK安装目录。
4. 若要执行样例：如果环境中存在多个版本python，将run.sh中python main.py修改为python3 main.py
```bash
bash run.sh
```

5. 若要进行性能测试：<br/>
需准备测试数据信息文档<br/>
测试信息文档至少包含：音频路径，音频时长，音频转录文本。json格式<br/>
例：{"wav_path":"./xx/xx/xxx.wav", "duration":5.292, "text":"上海也加入同比正增长的城市队列"}<br/>
修改data.yaml中data_info_dir为你的测试信息文档真实存放路径<br/>
根据你的实际情况修改performance_test.py中26-41行代码<br/>
将run.sh中python main.py修改为python performance_test.py<br/>
如果环境中存在多个版本python，将run.sh中python main.py修改为python3 performance_test.py

```bash
bash run.sh
```

## 6 其它说明

1. 此模型使用的数据集为[AISHELL数据集](http://www.aishelltech.com/kysjcp) 。<kbd>data/BAC009S0102W0436.wav</kbd>为其中一条语音，其对应的文字是：上海迪士尼度假区举办了一场媒体发布会。
2. 由于模型输入的限制，推理时wav语音的时长应控制在14.655s及其以下，超过14.655s的部分会被截断。
