# 模型推理工具

## 1 介绍
基于MindX SDK实现开发模型推理工具，用于测试om模型推理，本例为msame的python版本实现。   
[msame-C++工具链接](https://gitee.com/ascend/tools/tree/master/msame)

### 1.1 支持的产品

MindX SDK mxVision:2.0.4   
mxVision version:2.0.4.b096   
Plat: linux aarch64   

### 1.2 支持的版本

Ascend 21.0.4

### 1.3 适用场景   

使用于单输入或多输入模型推理   


### 1.4 代码目录结构与说明

```
.
├── img
│   ├── error1.jpg                            // msame-C++推理结果
│   │── error2.jpg                            // 本例(msame-python)推理结果
│   │── error3.jpg                            // 输入不一致报错截图
│   │── process.jpg                           // 流程图
├── set_env.sh                                // 需要设置的环境变量
├── Getnpy.py                                 // 生成npy文件示例代码
├── Getbin_yolov3.py                          // 生成yolov3的bin文件示例代码
├── Getbin_pointcnn.py                        // 生成pointcnn的bin文件示例代码
├── msame.py                                  // 模型推理工具代码
├── README.md                                 // ReadMe
```




### 1.5 技术实现流程图

![image-20220401173124980](./img/process.png)





## 2 环境依赖

推荐系统为ubuntu 18.04，环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| -------- | ------ |
| Python   | 3.9.0  |
| mxVision | 2.0.4  |
| numpy    | 1.21.2 |

软件依赖说明：

| 依赖软件 | 版本   | 说明                   |
| -------- | ------ | ---------------------- |
| numpy    | 1.21.2 | 将数据保存为二进制文件 |

在编译运行项目前，需要设置环境变量：

```
source set_env.sh
```

## 3 依赖安装

```
pip install numpy == 1.21.2
```

## 4 准备工作

[单输入模型yolov3样例]: 
[多输入模型pointnet样例]: 

注：多输入样例在输入时，多个输入用  ,  隔开。

准备工作：
yolov3输入npy文件生成可参考目录中Getnpy_yolov3.py，示例需在当前目录准备dog.jpg

```
python3.9 Getnpy_yolov3.py
```

转换成功后应在当前目录下产生dog.npy
yolov3输入bin文件生成可参考目录中Getbin_yolov3.py，示例需在当前目录准备dog.jpg

```
python3.9 Getbin_yolov3.py
```

转换成功后应在当前目录下产生dog.bin   

pointcnn输入bin文件生成可参考目录中Getbin_pointcnn.py   

```
python3.9 Getbin_pointcnn.py
```

转换成功后在当前目录下生成pointcnn.bin   

注：如上三个py文件分别适用于示例yolov3或poincnn的输入尺寸，更换模型需自行修改尺寸。   

##  5 编译与运行
示例步骤如下：   
**步骤1** 设置环境变量

**步骤2**  运行

```
python3.9 msame.py --model xxx --input xxx --output xxx  --loop xxx --outfmt xxx
```
在输出路径下成功输出预期的“.txt”或“.bin”则运行成功，否则报错。   
参数说明：
```
--input   
功能说明：模型的输入路径 参数值：bin或npy文件路径与文件名 示例：--input dog.npy
--output   
功能说明：模型的输出路径 参数值：bin或txt文件路径 示例：--output .
--model   
功能说明：om模型的路径 参数值：模型路径与模型名称 示例：--model yolov3.om
--outfmt    
功能说明：模型的输出格式 参数值：TXT 或 BIN 示例：--outfmt TXT
--loop   
功能说明：执行推理的次数 默认为1 参数值：正整数 示例：--loop 2
--device   
功能说明：执行代码的设备编号 参数值：自然数 示例：--device 1 
```
单输入模型：   
以yolov3模型npy文件输入作为示例参考：

```
python3.9 msame.py  --model yolov3_tf_bs1_fp16.om --input dog.npy --output test --outfmt TXT
```
执行成功后，在test目录下生成yolov3_tf_bs1_fp16_0.txt，yolov3_tf_bs1_fp16_1.txt， yolov3_tf_bs1_fp16_2.txt   
输出文件的个数与模型的输出有关。   

多输入模型：   

以pointcnn模型bin文件输入作为示例参考：    

```
python3.9 msame.py --model pointcnn_bs64.om --output test --outfmt BIN --input pointcnn.bin,pointcnn.bin
```

执行成功后，在test目录下生成pointcnn_bs64_0.bin     

## 6 常见问题
### 6.1 存储为txt格式时可能会出现第六位开始的误差，可以忽略此问题。  
 执行msame-C++输出结果：   
![image-20220401173124980](./img/error1.png)
 执行本例（msame-python）输出结果：   
![image-20220401173124980](./img/error2.png)
### 6.2 模型需要的输入与提供的数据不一致   
![image-20220401173124980](./img/error3.png)
问题描述：   
模型需要的输入shape与提供的数据shape不一致。   
解决方案：   
可在msame.py中自行输出m.input_shape查看模型需要的输入shape，m.input_dtype查看数据的shape。   
自行编写脚本查看bin或npy文件的shape和dtype，与模型的shape和dtype对齐。