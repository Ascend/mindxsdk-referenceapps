# 城市道路交通预测

## 1 介绍

STGCN主要用于交通预测领域，是一种时空卷积网络，解决在交通领域的时间序列预测问题。在定义图上的问题，并用纯卷积结构建立模型，这使得使用更少的参数能带来更快的训练速度。本样例基于MindxSDK开发，是在STGCN模型的基础上对SZ-Taxi数据集进行训练转化，可以对未来一定时段内的交通速度进行预测。通过在SZ-Taxi的测试集上进行推理测试，精度可以达到 MAE 2.81 | RMSE 4.29。该模型在SZ-Taxi数据集上无具体目标精度可以参考，此精度值为官方认可值。

论文原文：https://arxiv.org/abs/1709.04875

STGCN模型GitHub仓库：https://github.com/hazdzz/STGCN

SZ-Taxi数据集：https://github.com/lehaifeng/T-GCN/tree/master/data	

SZ-Taxi数据集包含深圳市的出租车轨迹，包括道路邻接矩阵和道路交通速度信息。

### 1.1 支持的产品
本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.2 软件方案介绍

基于MindX SDK的城市道路交通预测模型的推理流程为：

首先读取已有的交通速度数据集（csv格式）通过Python API转化为protobuf的格式传送给appsrc插件输入，然后输入模型推理插件mxpi_tensorinfer，最后通过输出插件mxpi_dataserialize和appsink进行输出。本系统的各模块及功能如表1.1所示：

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | 数据输入 | 调用pythonAPI的SendProtobuf()函数和MindX SDK的appsrc输入数据|
| 2    | 模型推理 | 调用MindX SDK的mxpi_tensorinfer对输入张量进行推理 |
| 3    | 结果输出 | 调用MindX SDK的mxpi_dataserialize和appsink以及pythonAPI的GetProtobuf()函数输出结果 |

### 1.3 特性及适用场景
模型的原始训练是基于SZ-Taxi数据集训练的，读取的图为深圳罗湖区156条主要道路的交通连接情况。因此对于针对罗湖区的自定义交通速度数据（大小为N×156，N>12），都能给出具有参考价值的未来一定时段的交通速度，从而有助于判断未来一段时间内道路的拥堵情况等。

### 1.4 代码目录结构与说明

eg：本sample工程名称为STGCN，工程目录如下图所示：
```
├── data                # 数据目录
├── stgcn10.om          # 转化得到的om模型
├── pipeline
│   └── stgcn.pipeline
├── main.py             # 展示推理精度
├── predict.py          # 根据输入的数据集输出未来一定时段的交通速度
├── README.md
├── convert_om.sh       # onnx文件转化为om文件
├── results             # 预测结果存放
└── train_need
    └── export_onnx.py  # 将pth文件转化成onnx文件，添加进训练项目
```

## 2 环境依赖

eg：推荐系统为ubuntu 18.04，环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| -------- | ------ |
| mxVision | 2.0.4 |
| Python | 3.9 |
| CANN | 5.1.RC1 |

- 环境变量介绍
在编译运行项目前，需要设置环境变量：
```
. ${SDK安装路径}/set_env.sh
. ${CANN安装路径}/set_env.sh
```

## 依赖安装
```
CANN软件包获取地址：https://www.hiascend.com/software/cann/commercial
SDK官方下载地址：https://www.hiascend.com/zh/software/mindx-sdk
我的安装步骤是本地下载好对应版本的安装包然后上传到服务器，然后再完成以下两个步骤
1、给.run安装包设置可执行权限
2、执行安装指令
./ *.run --install
```

## 3 城市道路交通预测开发实现
总体流程如下：
```
模型训练->模型转化->模型推理
```
### 3.1 模型训练
首先需要使用STGCN对SZ-Taxi数据集进行训练，使用的模型代码和数据集获取方式如下。
```
STGCN模型GitHub仓库：https://github.com/hazdzz/STGCN

SZ-Taxi数据集：https://github.com/lehaifeng/T-GCN/tree/master/data	
```
自行参照GitHub项目中的README.md和requiremments.txt文件配置训练所需环境。
为训练SZ-Taxi数据集，主要需要修改两个部分：
```
1、stgcn.py部分
（1）训练参数如下：
'learning_rate': 0.001,
'epochs': 1000,
'batch_size': 8,
'gamma': 0.95,
'drop_rate': 0.5,
'weight_decay_rate': 0.0005
（2）将数据集放到指定文件夹后增加
args.dataset == 'sz-taxis'

2、dataloader.py部分
（1）load_adj()
读取邻接矩阵部分改为
my_data = np.genfromtxt('data/sz-taxis/sz_adj.csv', delimiter=',') # 邻接矩阵路径
smy_data = sp.csr_matrix(my_data)
adj = smy_data.tocsc()
并且增加
elif dataset_name == 'sz-taxis':
        n_vertex = 156
（2）load_data()
train的划分改为：
train = vel[: len_train + len_val]
```
修改完毕后将SZ-Taxi数据集中的sz_adj.csv和sz_speed.csv文件放在'data/sz-taxis/'目录下（如果没有自行创建），运行STGCN模型GitHub仓库中的main.py文件即可开始训练。训练获得pth文件可通过export_onnx.py转换成onnx文件。

训练好的pth文件连接如下：
```
https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/STGCN/stgcn_sym_norm_lap_45_mins.pth
```

### 3.2 模型转化
本项目推理模型权重采用Github仓库中Pytorch框架的STGCN模型训练SZ-Taxi数据集得到的权重转化得到。经过以下两步模型转化：
1、pth转化为onnx
可以根据实际的路径和输入大小修改export_onnx.py（该文件需要依赖于项目结构目录，请放到训练代码项目主目录下再运行）
运行指令如下：
```
python export_onnx.py
```
转换好的onnx文件连接如下：
```
https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/STGCN/stgcn10.onnx
```

2、onnx转化为om
根据实际路径修改convert_om.sh
```
bash convert_om.sh [model_path] stgcn10
参数说明：
model_path：onnx文件路径须自行输入。
stgcn10：生成的om模型文件名，转换脚本会在此基础上添加.om后缀。
```

## 4 模型推理
### 4.1 pipeline编排
```
    appsrc # 输入
    mxpi_tensorinfer # 模型推理
    mxpi_dataserialize
    appsink # 输出
```
### 4.2 主程序流程

1、初始化流管理。
2、读取数据集。
3、向流发送数据，进行推理。
4、获取pipeline各插件输出结果。
5、销毁流。

## 5 运行
### 5.1 运行main.py
运行main.py可以在sz_speed.csv的测试集上获得推理精度，指令如下：
```
python main.py [image_path] [result_dir] [n_pred]

参数说明：
image_path：验证集文件，如“data/sz_speed.csv”
result_dir：推理结果保存路径，如“results/”
n_pred：预测时段，如9

例如： python main.py data/sz_speed.csv results/ 9
注意：sz_speed.csv文件的第一行数据为异常数据，需要手动删除
```
最后sz_speed.csv测试集的推理预测的结果会保存在results/predictions.txt文件中，实际数据会保存在results/labels.txt文件中。
推理精度会直接显示在界面上。
```
MAE 2.81 | RMSE 4.29
```
### 5.2 运行predict.py
如果需要推理自定义的数据集(行数大于12行，列数为156列的csv文件)，运行predict.py，指令如下：
```
python predict.py [image_path] [result_dir]

参数说明：
image_path：验证集文件，如“data/sz_speed.csv”
result_dir：推理结果保存路径，如“results/”

例如： python predict.py data/sz_speed.csv results/
```
则会在results文件夹下生成代表预测的交通速度数据prediction.txt文件
这是通过已知数据集里过去时段的交通速度数据预测未来一定时间内的交通速度，无标准参考，所以只会输出代表预测的交通速度数据的prediction.txt文件，而没有MAE和RMSE等精度。
另外和main.py的运行指令相比少一个n_pred参数，因为已在代码中定义了确定数值，无需额外输入。


## 6 常见问题
1、服务器上进行推理的时候出现coredump报错
```
原因：因为服务器上安装了好几个版本的mxVision，使用RC2版本的时候出现了这个问题，2.0.4版本的时候就可以了，是版本不匹配导致的。运行前可以先运行一下对应版本的set_env.sh
```