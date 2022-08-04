# BigGAN图像生成参考设计

## 1 介绍

   使用BigGAN模型，在MindxSDK环境下实现语义分割功能。

​   BigGAN模型的输入数据是由噪声数据和标签数据组成，其中噪声数据是由均值为0，方差为1的正态分布中采样，标签数据是由0至类别总数中随机采样一个整数。针对不同的batch size需要生成不同的输入数据。   

​   将随机生成的label标签和noise噪声数据，通过pipeline传入模型中进行推理，最终输出可视化图片。

注： 标签和噪声文件的生成需要在ModelZoo的ATC BIgGAN中下载并上传“biggan_preprocess.py”，“G_ema.pth”；在github仓中下载并上传“BigGAN.py”， “layers.py”， “inception_utils.py” 文件。运行如下命令，会在当前目录下生成“prep_label_bs1”和“prep_noise_bs1”文件夹，分别存放标签数据和噪声数据。命名逻辑从input_0000.bin开始依次递增。
    如何生成标签和噪声文件具体流程请参考第三节**前期数据和模型准备**。
```
python biggan_preprocess.py --batch-size 1 --num-inputs 1000
```



​  


### 1.1 支持的产品

​     本项目以昇腾Atlas310卡为主要的硬件平台

### 1.2 支持的版本

​     CANN：5.0.4（通过cat /usr/local/Ascend/ascend-toolkit/latest/acllib/version.info，获取版本信息）
​     SDK：2.0.4（可通过cat SDK目录下的version.info查看信息）


### 1.3 软件方案介绍

​        项目主要由主函数（main.py），数据集（prep_label_bs1, prep_noise_bs1），模型（biggan_sim_bs1.om），业务流（biggan.pipeline）组成。
​       主函数中构建业务流steam，先读取相应路径下的bin文件转换成pipeline可处理的tensor数据，再传入pipeline在sdk环境下先后实现模型推理的功能，最后从流中取出相应的输出数据转换数据类型保存结果。

​     表1.1 系统方案中各模块功能：

| 序号 | 模块        | 功能描述                                     |
| ---- | ----------- | -------------------------------------------- |
| 1    | appsrc      | 向Stream中发送数据，appsrc将数据发给下游元件 |
| 2    | tensorinfer | 对输入的张量进行推理                         |
| 3    | appsink     | 从stream中获取数据                           |
| 4    | saveImg     | 将数据保存成图像                             |


### 1.4 代码目录结构与说明

​     本工程名称为biggan，工程目录如下图所示：     

```
├── python  
|   ├── biggan.pipeline      //业务流
|   ├── main.py             // 主函数，用以在拥有数据集后生成图像
├── Readme.md    
├── SDK.jpg                 
└── RESULT.jpg
```

### 1.5 适用场景

​     工程使用了BigGAN网络模型，用以生成基于类别的合成图像。通过引入数据截断，正交正则化等技巧，能够在一定程度上提高图像多样性和真实性。

​            

## 2 环境依赖


​     推荐系统为ubantu 18.04，环境依赖软件和版本如下表：

| 软件名称            | 版本        | 说明                          | 获取方式                                                     |
| ------------------- | ----------- | ----------------------------- | ------------------------------------------------------------ |
| MindX SDK           | 2.0.4       | mxVision软件包                | [链接](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2FMindx-sdk) |
| ubuntu              | 18.04.1 LTS | 操作系统                      | Ubuntu官网获取                                               |
| Ascend-CANN-toolkit | 5.0.4       | Ascend-cann-toolkit开发套件包 | [链接](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fsoftware%2Fcann%2Fcommercial) |
| python              | 3.9.2       |                               |                                                              |
| numpy               | 1.22.4      | 维度数组运算依赖库            | 服务器中使用pip或conda安装                                   |
| opencv-python       | 4.6.0       | 图像处理依赖库                | 服务器中使用pip或conda安装                                   |



​     在编译运行项目前，需要设置环境变量：

​     . ${sdk_path}/set_env.sh

​     . ${ascend_toolkit_path}/set_env.sh

## 3.前期数据和模型准备

​     项目使用的模型为BigGAN，流程参考以下链接：

- ​     https://www.hiascend.com/zh/software/modelzoo/models/detail/1/c77dfa7e891f4388b62eeef7e8cbbc2d  

  请按照链接提示，完成数据集生成和模型文件的获取。

  

  ##### 注意事项：

1. 数据获取

   文件依赖问题：链接中执行“biggan_preprocess.py”脚本需要“BigGAN.py”， “layers.py”， “inception_utils.py” 来源于github。

   github文件下载路径：https://github.com/ajbrock/BigGAN-PyTorch.git 

   
   命名问题：样例中以bs为1，数据数量为1000为例，若改变了上述参数值，请对应调整网络模型命名以及main中num的数值。

2. 模型获取

​   直接获取已经训练并转好的om模型（biggan_sim_bs1.om）放在biggan/model目录下。如果没有model目录，需要用户自己新建一个文件夹并命名为model。

## 4.编译与运行



**步骤1** 进入biggan文件夹下：

```
cd biggan
```

**步骤2**  设置环境变量，如第2小节**环境依赖**所述。

**步骤3**   获取om模型，如第3小节**前期数据和模型准备**所述。若未从 pytorch 模型自行转换模型，使用的是上述链接提供的  om 模型，则无需修改相关文件，否则修改 python目录下pipeline的相关配置，将 mxpi_tensorinfer0 插件 modelPath 属性值中的 om 模型名改成实际使用的 om 模型名。

**步骤4**  进入python目录下：

```
cd python
```

**步骤5**  点击main.py，num设置为数据集的大小，count设置为需要转换图像的编号（默认为num为1000，count为11）。

**步骤6**  保存脚本文件，在命令行输入：

```
python main.py
```

**步骤7**   结果无误时会在biggan目录下生成result文件夹，文件夹中保存了count_result.jpg格式的生成图像。如下图所示：



![3](RESULT.jpg)

注： 具体实现图片依赖于label标签和noise噪声文件，文件随机生成，最终生成图片因人而异。

## 5.常见问题

**问题描述：**

提示 No module named 'MxpiDataType_pb2'

![1](SDK.jpg)

**解决方案：**

未配置SDK环境变量，可以输入命令查看环境变量设置是否正确。

```
env list
```



