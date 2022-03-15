## SDK 遥感影像地块分割检测样例

## 1 介绍
使用DANet和Deeplabv3+，其中两模型均使用了pytorch官方提供的resnet101预训练模型作为backbone,使用SGDR对模型进行训练,选择多个局部最优点的结果进行集成,输出输入图像的语义分割图 \
样例输入：一张256*256的遥感影像地图\
样例输出：输入图片的语义分割图

### 1.1 支持的产品
昇腾310(推理)

### 1.2 支持的版本
本样例配套的CANN版本为 [5.0.4](https://www.hiascend.com/software/cann/commercial) ，MindX SDK版本为 [2.0.4](https://www.hiascend.com/software/Mindx-sdk) 。

### 1.3 目录结构
```
|--RemoteSensingSegmentation
|-------- config
|           |---- configure.cfg                     // 模型转换配置文件
|-------- models                                    // 模型存放文件目录(自行创建)
|-------- pipeline
|           |---- segmentation.pipeline             // 遥感影像地块分割的pipeline文件
|-------- result                                    // 语义分割结果存放处(自行创建)
|           |---- final                             // 对比结果图存放目录,对比结果图（左为输入原图 右为结果图）
|           |---- temp_result                       // 单一结果图存放目录,仅有单一结果图
|-------- test_set                                  // 测试集图像目录(自行创建)
|           |---- *.jpg                             // 15张jpg格式的测试集遥感图片
|-------- util
|           |---- model_conversion.sh               // 模型转换脚本 *.onxx -> *.om
|           |---- transform_model_util.py           // 模型转换工具 *.pth -> *.onxx
|           |---- visual_utils.py                   // 语义分割可视化工具
|-------- main.py                                   // 遥感影像地块分割检测样例
|-------- README.md                                 // ReadMe 
```

## 2 模型转换

### 2.1 准备工作

**步骤1** 下载预训练模型权重文件 `37_Deeplabv3+_0.8063.pth`和`84_DANet_0.8081.pth`[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/RemoteSensingSegmentation/models.zip)

**步骤2** 将获取到的`37_Deeplabv3+_0.8063.pth`和`84_DANet_0.8081.pth`分别转换为 `Deeplabv3.onnx`和`DANet.onnx`(步骤1链接中已包含转换后的`onxx`格式模型，可直接跳到步骤2.5)

**步骤2.1** 需要在原项目模型源码基础上进行`*.pth->*.onxx`转换, 所以思路是在`PyCharm`中把项目克隆下来并安装环境，运行转换工具。首先使用 git 克隆[GitHub项目](https://github.com/JasmineRain/NAIC_AI-RS/blob/master) 到本地

**步骤2.2** 安装`Anaconda`并创建一个虚拟环境,且`pip install -r requirements.txt`安装项目所需环境依赖, 并额外安装依赖`onnx == 1.8.0`
```bash
conda create -n 虚拟环境名 python=3.9.2
conda activate 虚拟环境名
cd 克隆项目的目录
# 运行前修改requiremenst.txt版本:
#   改四个地方：mkl-random==1.2.1, tensorboard-plugin-wit==1.7.0, torch==1.8.0, torchvision==0.9.0
#   新增一行：在文件末增加 onnx==1.8.0, 额外安装依赖onnx
pip install -r requiremenst.txt  #安装依赖,可能会因为网络问题而报错,多运行几次
# 安装好后在pycharm工具里面选择使用创建的虚拟环境(settings->Python Interpreter)
```
**步骤2.3** 把`37_Deeplabv3+_0.8063.pth`和`84_DANet_0.8081.pth`复制到克隆项目`models`目录下

**步骤2.4** 把`transform_model_util.py`复制到克隆项目`根目录`下并执行

**步骤2.5** 步骤2.4执行完成后会在克隆项目`models`目录下生成`DANet.onnx`和`Deeplabv3.onnx`,复制到本项目`models`下执行步骤3

**步骤3** 在本项目`util`目录下运行模型转换脚本 `./model_conversion.sh`(如不是可执行脚本，先将其转换`chmod +x model_conversion.sh`), 执行完模型转换脚本后，会在`models`目录下生成相应的DANet和Deeplabv3的`om`模型文件

模型转换使用了ATC工具，如需更多信息请参考:

 https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html

### 2.2 pipeline配置

**步骤1** [segmentation.pipeline](./pipeline/segmentation.pipeline) 中配置 `DANet.om`和`Deeplabv3.om`模型路径（已配置）
```
"mxpi_tensorinfer0": {
    "props": {
        "dataSource": "mxpi_imageresize0",
        "modelPath": "models/DANet.om"
    },
    "factory": "mxpi_tensorinfer",
    "next": "mxpi_tensorinfer1"
},
"mxpi_tensorinfer1": {
    "props": {
        "dataSource": "mxpi_imageresize0",
        "modelPath": "models/Deeplabv3.om"
    },
    "factory": "mxpi_tensorinfer",
    "next": "appsink0"
},
```

## 3 模型推理

### 3.1 配置MindXSDK和Python环境变量
```bash
# 执行如下命令，打开.bashrc文件
cd $HOME
vi .bashrc
# 在.bashrc文件中添加以下环境变量
MX_SDK_HOME= SDK安装路径
LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
PYTHONPATH=${MX_SDK_HOME}/python

# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

# 查看环境变量
env
```
### 3.2 运行
> 运行前请下载 [测试集图片](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/RemoteSensingSegmentation/data.zip) 放入目录`test_set`下, 随后进入工程目录，键入执行指令
```bash
# 进入工程目录
cd RemoteSensingSegmentation

# 图片测试集在test_set目录下，一共15张遥感地图
python3 main.py ${测试图片路径} ${是否开启对比图输出} ${输出结果路径}
e.g.: python3 main.py test_set/test_1.jpg True result/final/result.jpg
```

### 3.3 查看结果
```
运行完毕后, 如果开启了对比图输出, 对比图结果保存在工程目录result/final下中, 单一结果图保存在工程目录result/temp_result下，若没有开启对比图输出，仅有单一结果图输出
```