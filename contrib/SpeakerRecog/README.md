# Speaker Recognition

## 1 简介
  本开发样例基于MindX SDK实现了说话人识别（Speaker Recognition）。<br/>
  Speaker Recognition主要分为三个步骤：<br/>

  **一、 构建声纹模型** <br/>
  **二、 说话人注册（提取说话人embedding，保存到声纹库中）** <br/>
  **三、 说话人识别（提取说话人embedding, 与声纹库中的声纹作相似度判别）** <br/>

  声纹模型采用论文《X-vectors: Robust DNN Embeddings for Speaker Recognition》提出的x-vector。<br/>
  模型搭建参考 https://github.com/manojpamk/pytorch_xvectors/blob/master/models.py <br/>
  在本样例中，注册、识别流程如下：声纹库为空时，直接对当前说话人进行注册（注册名使用当前文件名）。声纹库不为空时 <br/>
  进行说话人识别，如果声纹库中不包含当前说话人，对当前说话人进行注册，否则给出识别结果。<br/>

## 2 模型转换

由于原模型是onnx模型，需借助ACT工具将onnx模型转换为om模型。模型转换时应先按照[准备动作](https://support.huaweicloud.com/atc-model-convert-cann202infer/atlasatc_16_0005.html)  
配置好环境和设置环境变量，然后执行以下命令
```bash
atc --framework=5 --model={model_path} --input_shape="fbank:1,64,1000"
    --output=x_vector_batch_one  --soc_version=Ascend310
```


## 3 目录结构


```
.
|-------- model
|           |---- x_vector         
|                     |---- x_vector_batch_one.om        //声纹模型
|-------- pipeline
|           |---- SpeakerRecog.pipeline        //声纹模型流水线配置文件
|-------- python
|           |---- main.py                      //测试样例
|           |---- performance_test.py          //模型性能测试
|           |---- post_process.py              //说话人注册、识别
|           |---- preprocessing.py             //对语音数据进行特征提取
|           |---- run.sh                       //样例运行脚本
|           |---- utils.py                     //性能测试所需函数
|-------- test_wav                             //用于测试的语音样本
|-------- voice_print_library                  //声纹库
|-------- README.md
```

## 4 依赖

|软件名称    | 版本     |
|-----------|----------|
| python    | 3.7.5   |
| MindX SDK | 2.0.2    |
| librosa   | 0.8.1    |
| torch     | 1.9.0    |
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
4. 若要执行样例：
如果环境中存在多个版本python，将run.sh中python main.py修改为python3 main.py
```bash
bash run.sh
```
如果出现 XXX registration complete! 说明注册成功<br/>
如果出现The current audio XXX.wav  is from speaker XXX 说明识别成功<br/>
5. 若要进行性能测试：<br/>
修改performance_test.py中数据集路径以及embedding存放路径<br/>
将run.sh中python main.py修改为python performance_test.py<br/>
如果环境中存在多个版本python，将run.sh中python main.py修改为python3 performance_test.py
```bash
bash run.sh
```

## 6 其它说明

1. 此模型使用的数据集为[AISHELL数据集](http://www.aishelltech.com/kysjcp) 。
2. 由于模型输入的限制，wav语音的时长应控制在10s及其以下，超过10s的部分会被截断。
