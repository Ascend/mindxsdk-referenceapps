# MMNET人像分割

## 1 介绍
MMNet致力于解决移动设备上人像抠图的问题，旨在以最小的模型性能降级在移动设备上获得实时推断。MMNet模型基多分支dilated conv以及线性bottleneck模块，性能优于最新模型，并且速度提高了几个数量级。

本开发样例基于MindX SDK实现人像分割的功能，其主要功能是利用MMNET模型对输入图片中的人像进行灰度提取，从而与背景分离开，生成一张人像分割图片。

样例输入：带有人体的jpg图片

样例输出：人像与背景分离的新图片

### 1.1 支持的产品

项目所用的硬件平台：Ascend310

### 1.2 支持的版本

支持的SDK版本为 2.0.4，CANN版本为 20.2.0 

### 1.3 代码目录结构与说明

本工程名称为MMNET，工程目录如下图所示：

```
|-------- test                                // 存放测试图片
|-------- mask                                // 存放测试图片mask掩膜
|-------- model
|           |---- mmnet.aippconf              // aipp配置文件
|-------- main.py                             // 主程序  
|-------- pipeline                               
|           |---- MMNET.pipeline              // pipeline流水线配置文件 
|-------- evaluate.py                         // 精度测试程序
|-------- README.md   
```

### 1.4 场景限制

本项目能够针对人像清晰的图像完成人像分割任务并实现可视化。对于大部分人像图片，在图像清晰且人像在图片中占据较大比例的情况下都可以进行正确识别。但由于MMNET原算法的局限性，在部分情况下识别效果较差，具体如下：

1.人像在图片中比例过小，会出现漏检的情况；

2.对于存在多张人物目标的图片，会无法正确识别并分割；

3.对于环境杂乱颜色过多的图片，分割效果较差，建议使用纯色的背景。

建议使用纯色的背景，且目标在图片中占比较大的图片进行测试。

## 2 环境依赖

| 软件名称  | 版本  |
| --------- | ----- |
| MindX SDK | 2.0.4 |
| python    | 3.9.2   |
| CANN      | 5.0.4  |
| opencv2   |       |
| numpy     |       |


在编译运行项目前，需要设置环境变量：

- 环境变量介绍
运行以下设置脚本以完成，其中{%Mind_SDK%}请替换为实际SDK安装位置

```
source {%Mind_SDK%}/mxVision-3.0.RC3/
```



## 3 模型转换
人像分割采用提供的mmnet.pb模型。由于原模型是基于tensorflow的人像分割模型，因此我们需要借助于ATC工具将其转化为对应的om模型。

具体步骤如下：
**步骤1** 获取模型pb文件
，下载链接为https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MMNET/mmnet.pb

**步骤2** 将获取到的mmnet模型pb文件存放至：“项目所在目录/model”

**步骤3** 模型转换

在pb文件所在目录下存放aipp配置文件，文件名为，具体内容如下：

```python
aipp_op {
    aipp_mode: static
    src_image_size_w :256
    src_image_size_h :256
    input_format : YUV420SP_U8
    csc_switch : true
    rbuv_swap_switch : false
    matrix_r0c0 : 256
    matrix_r0c1 : 0
    matrix_r0c2 : 359
    matrix_r1c0 : 256
    matrix_r1c1 : -88
    matrix_r1c2 : -183
    matrix_r2c0 : 256
    matrix_r2c1 : 454
    matrix_r2c2 : 0
    input_bias_0 : 0
    input_bias_1 : 128
    input_bias_2 : 128
}
```

在确保环境变量设置正确后，在pb文件所在目录下执行以下命令：

```
atc --model=mmnet.pb --framework=3 --output=mmnet --soc_version=Ascend310 --insert_op_conf=mmnet.aippconf --input_shape="input_x:1,256,256,3"
```

执行完模型转换脚本后，若提示如下信息说明模型转换成功，会在output参数指定的路径下生成mmnet.om模型文件。

```python
ATC run success  
```



## 4 编译运行

接下来进行模型的安装运行，具体步骤如下：

**步骤1** 获取om模型

**步骤2** 修改run.sh中MX_SDK_HOME


**步骤3** 配置pipeline

根据所需场景，配置pipeline文件，调整路径参数等。

```python
"mxpi_tensorinfer0": {
			"props": {
				"modelPath": "./model/mmnet.om"
			},
			"factory": "mxpi_tensorinfer",
			"next":"appsink0"
#修改om文件存放的路径
```

**步骤4** 存放图片，执行模型进行测试

将测试图片存放至主目录下，修改main.py中的图片存放路径以及人像分割后的存储路径的相关代码：
【注意】测试图片尽量仅包含一个人物，正脸且周围环境较为简单，同时图片为jpg格式。否则会对人像分割效果有较大影响,造成较大误差。

```
filepath = "test.jpg"
filepath_out = "test-out.jpg"
```

然后执行run.sh文件：

```
bash run.sh
```

输出的图片即为样例的人像分割后的图片。

## 5 精度测试

对测试集中的300张图片进行精度测试，具体步骤如下：

**步骤1** 获取测试集的图片,确保测试集的输入图片为jpg格式。
获取地址为：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MMNET/data.zip

**步骤2** 修改evaluate.py中的测试集图片存放路径：

```
filepath = "./test/"  #测试集图片存放路径
gt_dir = './mask'   #测试集掩膜mask图片存放路径
```

**步骤3** 修改run.sh中MX_SDK_HOME和执行文件名称：

```
python3 evaluate.py
```

并执行：

```
bash run.sh
```

