# MindXSDK 车辆姿态识别

## 1 简介
本开发样例基于MindX SDK实现了姿态估计网络(PoseEstNet)，用于检测并预测车辆36个关键点坐标以及生成热点图。

## 2 目录结构
本工程名称为PoseEstNet，工程目录如下图所示：

```
PoseEstNet
|---- models
|   |   |---- aipp_nv12.cfg
|   |   |---- coco.names
|   |   |---- PoseEstNet.om
|   |   |---- yolov3.cfg
|   |   |---- yolov3.om
|   |   |---- aipp_hrnet_256_256.aippconfig
|---- pipeline                          // 流水线配置文件夹
|   |   |---- eval_PoseEstNet.pipeline
|   |   |---- PoseEstNet.pipeline
|---- plugins                           // 插件文件夹
|---- data                              
|---- data_eval 
|   |   |---- images
|   |   |---- labels
|---- output                            // 结果保存文件夹                              
|---- output_eval       
|---- main.py
|---- eval.py
|---- README.md   
```

## 3 依赖
| 软件名称 | 版本   |
| :--------: | :------: |
|ubuntu 18.04|18.04.1 LTS   |
|CANN|5.0.4|
|MindX SDK|2.0.4|
|Python| 3.9.12|
|numpy | 1.22.4 |
|opencv_python|4.6.0|  
|cmake|3.5+| 
请注意MindX SDK使用python版本为3.9.12，如出现无法找到python对应lib库请在root下安装python3.9开发库  
```
apt-get install libpython3.9
```

## 4 模型转换
车辆姿态识别先采用了yolov3模型将图片中的车辆检测出来，然后利用PoseEstNet模型预测车辆36个关键点坐标。

4.1 yolov3的模型转换：  

**步骤1** 获取yolov3的原始模型(.pb文件)和相应的配置文件(.cfg文件)  
&ensp;&ensp;&ensp;&ensp;&ensp; [原始模型下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PoseEstNet/yolov3_tensorflow_1.5.pb)
&ensp;&ensp;&ensp;&ensp;&ensp; [配置文件下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PoseEstNet/aipp_nv12.cfg)  

**步骤2** 将获取到的yolov3模型.pb文件和.cfg文件存放至：“项目所在目录/models”  

**步骤3** .om模型转换  
以下操作均在“项目所在目录/models”路径下进行：  
- 设置环境变量，进入根目录，执行如下命令后返回原目录
```
source usr/local/Ascend/ascend-toolkit/set_env.sh
```
- 使用ATC将.pb文件转成为.om文件
```
atc --model=yolov3_tensorflow_1.5.pb --framework=3 --output=yolov3 --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0" --log=info --insert_op_conf=aipp_nv12.cfg
```
- 执行完模型转换脚本后，若提示如下信息说明模型转换成功，可以在该路径下找到名为yolov3.om模型文件。
（可以通过修改output参数来重命名这个.om文件）
```
ATC run success, welcome to the next use.
```  

4.2 PoseEstNet的模型转换

4.2.1 模型概述  
&ensp;&ensp;&ensp;&ensp;&ensp; [PoseEstNet论文地址](https://arxiv.org/pdf/2005.00673.pdf)
&ensp;&ensp;&ensp;&ensp;&ensp; [PoseEstNet代码地址](https://github.com/NVlabs/PAMTRI/tree/master/PoseEstNet)

4.2.2 模型转换环境需求
```
- 框架需求
  CANN == 5.0.4
  torch == 1.5.0
  torchvision == 0.6.0
  onnx == 1.11.1

- python第三方库
  numpy == 1.22.4
  opencv-python == 4.6.0
  Pillow == 8.2.0
  yacs == 0.1.8
  pytorch-ignite == 0.4.5
```

4.2.3 模型转换步骤

**步骤1** .pth模型转.onnx模型

***1*** 获取.pth权重文件：&ensp;models/veri/pose_hrnet/w32_256x256_adam_lr1e-3/model_best.pth
```
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1vD08fh-za3mgTJ9UkK1ASCTJAqypW0RL' -O models.zip
unzip models.zip
rm models.zip
```
[Huawei Cloud下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PoseEstNet/model_best.pth)

***2*** 获取PoseEstNet_pth2onnx.py 
&ensp; 下载PoseEstNet源码并创建项目，将该脚本放在“项目所在目录/models”路径下，执行下列命令，生成.onnx模型文件
```
python3 tools/PoseEstNet_pth2onnx.py --cfg experiments/veri/hrnet/w32_256x256_adam_lr1e-3.yaml TEST.MODEL_FILE models/veri/pose_hrnet/w32_256x256_adam_lr1e-3/model_best.pth
```
> 注意目前ATC支持的onnx算子版本为11  

此时在“项目所在目录/models”路径下会出现PoseEstNet.onnx模型，到此步骤1已完成  
如果在线环境中无法安装pytorch，你可以在本地环境中进行上述.pth模型转.onnx模型操作，然后将得到的.onnx模型放在“项目所在目录/models”即可

本项目提供onnx模型：[Huawei Cloud下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PoseEstNet/PoseEstNet.onnx)

**步骤2** .onnx模型转.om模型

***1*** 设置环境变量，进入根目录，执行如下命令后返回原目录
```
source usr/local/Ascend/ascend-toolkit/set_env.sh
```

***2*** 进入.onnx文件所在目录，使用ATC将.onnx文件转成为.om文件(注意文件路径)
```
atc --framework=5 --model=PoseEstNet.onnx --output=PoseEstNet --input_format=NCHW --input_shape="image:1,3,256,256" --insert_op_conf=aipp_hrnet_256_256.aippconfig --log=debug --soc_version=Ascend310
```
- 执行完模型转换脚本后，若提示如下信息说明模型转换成功（同样的，可以通过修改output参数来重命名这个.om文件）
```
ATC run success, welcome to the next use.
```  

经过上述操作，可以在“项目所在目录/models”找到yolov3.om模型和PoseEstNet.om模型，模型转换操作已全部完成

4.3 参考链接
> 模型转换使用了ATC工具，如需更多信息请参考：[ATC工具使用指南-快速入门](https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html)

## 5 数据集  
5.1 原始VeRi数据集  

&ensp;&ensp;&ensp;&ensp;&ensp; [Github下载链接](https://vehiclereid.github.io/VeRi/)
原数据集images文件夹下面分为images_train和images_test，需要自己将这两个文件夹里的图片复制到data_eval/images文件夹下面
目录结构如下：
```
├── data_eval
    ├── images
    |   ├── 0010_c014_00034990_0.jpg
    |   ├── 0010_c017_00034050_1.jpg
    ├── labels
    |   ├── label_test.csv
```
其中data/labels中的csv文件：[Github下载链接](https://github.com/NVlabs/PAMTRI/tree/master/PoseEstNet/data/veri/annot)

----------------------------------------------------
## 6 测试

6.1 配置环境变量  

运行cann和sdk的set_env.sh脚本

6.2 获取om模型
```
步骤详见4： 模型转换
```
6.3 准备数据集
```
步骤详见5： 数据集
```
6.4 安装插件编译所需要的NumCpp库
```
cd plugins
git clone https://github.com/dpilger26/NumCpp
mkdir include
cp -r  NumCpp/include/NumCpp ./include/
```
6.5 编译插件
```
bash build.sh
```
6.6 切换到插件目录更新权限
```
cd ${MX_SDK_HOME}/lib/plugins
```
将libmxpi_pnetpostprocessplugin.so和libmxpi_pnetpreprocessplugin.so权限更改为640

6.7 切换回根目录，创建文件夹
```
cd (项目根目录)
mkdir data
mkdir data_eval
mkdir output
mkdir output_eval
```
6.8 配置pipeline  
根据所需场景，配置pipeline文件，调整路径参数等。

PoseEstNet.pipeline:
```
    # 配置mxpi_tensorinfer插件的yolov3.om模型加载路径（lines 26-33）
    lines 26-33:
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "models/yolov3.om(这里根据你的命名或路径进行更改)"
            },
            "factory": "mxpi_tensorinfer",
            "next": "mxpi_objectpostprocessor0"
        },
    # 配置mxpi_objectpostprocessor插件的yolov3.cfg配置文件加载路径以及SDN的安装路径（lines 34-43）
    lines 34-43:
        "mxpi_objectpostprocessor0": {
           "props": {
                    "dataSource": "mxpi_tensorinfer0",
                    "postProcessConfigPath": "models/yolov3.cfg(这里根据你的命名或路径进行更改)",
                    "labelPath": "models/coco.names",
                    "postProcessLibPath": "${SDK安装路径}/lib/modelpostprocessors/libyolov3postprocess.so"
                },
                "factory": "mxpi_objectpostprocessor",
                "next": "mxpi_imagecrop0"
        },
    # 配置mxpi_tensorinfer插件的PoseEstNet.om模型加载路径（lines 68-75 以及 92-99）
    lines 68-75：
        "mxpi_tensorinfer2":{
            "props": {
                "dataSource": "mxpi_preprocess1",
                "modelPath": "models/PoseEstNet.om(这里根据你的命名或路径进行更改)"
            },
            "factory":"mxpi_tensorinfer",
            "next":"mxpi_postprocess1"
        },

```
eval_PoseEstNet.pipeline:
```
 # 配置mxpi_tensorinfer插件的PoseEstNet.om模型加载路径（lines 28-35）
    lines 28-35：
        "mxpi_tensorinfer2":{
            "props": {
                "dataSource": "mxpi_preprocess1",
                "modelPath": "models/PoseEstNet.om(这里根据你的命名或路径进行更改)"
            },
            "factory":"mxpi_tensorinfer",
            "next":"mxpi_postprocess1"
        },
```

6.9 执行

业务代码main.py结果在output文件夹
```
python3 main.py --inputPath data
```
评估代码的具体结果在output_eval文件夹
```
python3 eval.py --inputPath data_eval/images/ --labelPath data_eval/labels/label_test.csv 
```


## 7 精度对比

项目精度：
![项目精度](image/output_eval.png)

目标精度：

| Train set | VeRi   |
| :--------: | :------: |
|Wheel|85.10   |
|Fender|81.14|
|Back SDK|69.20|
|Front| 77.44|
|WindshieldBack | 85.67 |
|WindshieldFront|89.92|  
|Mean|82.15| 
