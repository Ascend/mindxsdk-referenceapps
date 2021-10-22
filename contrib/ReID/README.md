# MindXSDK 行人重识别

## 1 简介
本开发样例基于MindX SDK实现了端到端的行人重识别（Person Re-identification, ReID），支持检索给定照片中的行人ID。其主要流程为：    
- 程序入口分别接收查询图片和行人底库所在的文件路径。    
- 对于查询图片：利用目标检测模型YOLOv3推理，检测图片中的行人，检测结果经过抠图与调整大小，再利用ReID模型提取图片中每个行人的特征向量。    
- 对于行人底库：将底库图片调整大小，利用ReID模型提取相应的特征向量。    
- 行人检索：将查询图片中行人的特征向量与底库中的特征向量，为每个查询图片中的行人检索最有可能的ID，通过识别框和文字信息进行可视化标记。

## 2 目录结构
本工程名称为ReID，工程目录如下图所示：
```
ReID
|---- data
|   |---- gallerySet                    // 查询场景图片文件夹
|   |---- querySet                      // 行人底库图片文件夹
|   |---- ownDataset					// 自制行人底库原图形文件夹
|   |---- cropOwnDataset				// 自制行人底库结果文件夹
|---- models                            // 目标检测、ReID模型与配置文件夹
|   |   |---- yolov3.cfg
|   |   |---- coco.names
|   |   |---- ReID_pth2onnx.py
|   |   |---- ReID_pth2onnx.cfg
|---- pipeline                          // 流水线配置文件夹
|   |   |---- ReID.pipeline
|---- result                            // 结果保存文件夹                              
|---- main.py       
|---- makeYourOwnDataset.py
|---- README.md   
```
> 由于无法在Gitee上创建空文件夹，请按照该工程目录，自行创建data文件夹、result文件夹以及其内部的文件夹  
> 如果没有创建result文件夹，将无法产生输出  
## 3 依赖
| 软件名称 | 版本   |
| :--------: | :------: |
|ubantu 18.04|18.04.1 LTS   |
|MindX SDK|2.0.2|
|Python| 3.7.5|
|numpy | 1.21.0 |
|opencv_python|4.5.2|  
请注意MindX SDK使用python版本为3.7.5，如出现无法找到python对应lib库请在root下安装python3.7开发库  
```
apt-get install libpython3.7
```
## 4 模型转换
行人重识别先采用了yolov3模型将图片中的行人检测出来，然后利用ReID模型获取行人的特征向量。由于yolov3模型和ReID模型分别是基于Pytorch和Tensorflow的深度模型，我们需要借助ATC工具分别将其转换成对应的.om模型。

4.1 yolov3的模型转换：  

**步骤1** 获取yolov3的原始模型(.pb文件)和相应的配置文件(.cfg文件)  
&ensp;&ensp;&ensp;&ensp;&ensp; [原始模型下载链接](https://c7xcode.obs.myhuaweicloud.com/models/YOLOV3_coco_detection_picture_with_postprocess_op/yolov3_tensorflow_1.5.pb)
&ensp;&ensp;&ensp;&ensp;&ensp; [配置文件下载链接](https://c7xcode.obs.myhuaweicloud.com/models/YOLOV3_coco_detection_picture_with_postprocess_op/aipp_nv12.cfg)  

**步骤2** 将获取到的yolov3模型.pb文件和.cfg文件存放至：“项目所在目录/models”  

**步骤3** .om模型转换  
以下操作均在“项目所在目录/models”路径下进行：  
- 设置环境变量（请确认install_path路径是否正确）
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest    

export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
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

4.2 ReID的模型转换

4.2.1 模型概述  
&ensp;&ensp;&ensp;&ensp;&ensp; [ReID论文地址](https://arxiv.org/pdf/1903.07071.pdf)
&ensp;&ensp;&ensp;&ensp;&ensp; [ReID代码地址](https://github.com/michuanhaohao/reid-strong-baseline)

4.2.2 模型转换环境需求
```
- 框架需求
  CANN == 5.0.1
  torch == 1.5.0
  torchvision == 0.6.0
  onnx == 1.7.0

- python第三方库
  numpy == 1.21.0
  opencv-python == 4.5.2
  Pillow == 8.2.0
  yacs == 0.1.8
  pytorch-ignite == 0.4.5
```

4.2.3 模型转换步骤

**步骤1** .pth模型转.onnx模型  

***1*** 从GitHub上拉取ReID模型源代码,在“项目所在目录/models”路径下输入：  
```
git clone https://github.com/michuanhaohao/reid-strong-baseline
```
此时会出现“项目所在目录/models/reid-strong-baseline”路径，路径内是ReID模型的源代码  

***2*** 获取.pth权重文件，将该.pth权重文件放在“项目所在目录/models”路径下  
文件名：market_resnet50_model_120_rank1_945.pth  
&ensp;&ensp;&ensp;&ensp;&ensp; [Google Drive](https://drive.google.com/drive/folders/1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A)
&ensp;&ensp;&ensp;&ensp;&ensp; [Baidu Cloud, 提取码: v5uh](https://pan.baidu.com/s/1ohWunZOrOGMq8T7on85-5w)

***3*** 获取ReID_pth2onnx.py：[获取链接](https://gitee.com/ascend/modelzoo/blob/master/contrib/ACL_PyTorch/Research/cv/classfication/ReID_for_Pytorch/ReID_pth2onnx.py)  
&ensp; 将该脚本放在“项目所在目录/models”路径下，执行下列命令，生成.onnx模型文件
```
python3.7 ReID_pth2onnx.py --config_file='reid-strong-baseline/configs/softmax_triplet_with_center.yml' MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('market_resnet50_model_120_rank1_945.pth')"
```
> 注意目前ATC支持的onnx算子版本为11  

此时在“项目所在目录/models”路径下会出现ReID.onnx模型，到此步骤1已完成  
如果在线环境中无法安装pytorch，你可以在本地环境中进行上述.pth模型转.onnx模型操作，然后将得到的.onnx模型放在“项目所在目录/models”即可


**步骤2** .onnx模型转.om模型

***1*** 设置环境变量
> 请重复一次4.1中步骤3的“设置环境变量（请确认install_path路径是否正确）”操作

***2*** 使用ATC将.onnx文件转成为.om文件
```
atc --framework=5 --model=ReID.onnx --output=ReID --input_format=NCHW --input_shape="image:1,3,256,128" --insert_op_conf=ReID_onnx2om.cfg --log=debug --soc_version=Ascend310
```
- 执行完模型转换脚本后，若提示如下信息说明模型转换成功，可以在“项目所在目录/models”路径下找到名为ReID.om模型文件。（同样的，可以通过修改output参数来重命名这个.om文件）
```
ATC run success, welcome to the next use.
```  

经过上述操作，可以在“项目所在目录/models”找到yolov3.om模型和ReID.om模型，模型转换操作已全部完成

4.3 参考链接
> 模型转换使用了ATC工具，如需更多信息请参考：[ATC工具使用指南-快速入门](https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html)  
> Yolov3模型转换的参考链接：[ATC_yolov3_tensorflow](https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/yolov3/ATC_yolov3_tensorflow/)  
> ReID模型转换的参考链接：[ReID_for_Pytorch](https://gitee.com/ascend/modelzoo/tree/master/contrib/ACL_PyTorch/Research/cv/classfication/ReID_for_Pytorch/#31-pth%E8%BD%AConnx%E6%A8%A1%E5%9E%8B)  

## 5 数据集  
5.1 Market1501数据集  

文件名：Market-1501-v15.09.15.zip  
&ensp;&ensp;&ensp;&ensp;&ensp; [Google Drive](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ)
&ensp;&ensp;&ensp;&ensp;&ensp; [Baidu Cloud](https://pan.baidu.com/s/1ntIi2Op)

5.1.1 行人底库  
请解压“Market-1501-v15.09.15.zip”文件，在“Market-1501-v15.09.15\Market1501\gt_bbox”中选择想要查询的行人图片，将图片放在“项目所在目录/data/querySet”中  
> 推荐每次查询1人，使用2-6张图片作为底库，效果较好  
> 如需要查询多人，请保证待查询行人之间的着装风格差异较大，否则会较容易出现误报  
> 该项目需要为每张图片提取行人ID，行人图片的命名格式为  
>> '0001(行人ID)_c1(相机ID)s1(录像序列ID)_000151(视频帧ID)_00(检测框ID).jpg'

5.1.2 场景图片数据集  
这里使用的是market1501中的部分场景图片数据，来源于
[Person Search Demo](https://github.com/songwsx/person_search_demo/tree/master/data/samples)
，也可以通过[Baidu Cloud，提取码：2xm1](https://pan.baidu.com/s/1UcqZ0G7X8dejR8ROPMsU5A)
获取，然后将获取的图片放在“项目所在目录/data/gallerySet”中 

5.2 自制数据集  
这里需要注意的是，自制数据集中的所有图片必须严格控制为横屏风格（图片的长度必须严格大于宽度）  
涉及文件夹  
> “项目所在目录/data/ownDataset”：用于存放制作行人底库的场景图片  
> “项目所在目录/data/cropOwnDataset”：用于保存从场景图片提取的行人图片  
  
**步骤1** 请将所有的场景图片分成不相交的两个部分：  
> 一个部分用于制作行人底库（放在“项目所在目录/data/ownDataset”路径下）  
> 另一个部分用于查询（放在“项目所在目录/data/gallerySet”路径下）

**步骤2** 调用makeYourOwnDataset.py将“项目所在目录/data/ownDataset”路径下场景图片中的所有行人提取出来，结果存放在“项目所在目录/data/cropOwnDataset”中
```
python3.7 makeYourOwnDataset.py --imageFilePath='data/ownDataset' --outputFilePath='data/cropOwnDataset'
```
**步骤3** 根据“项目所在目录/data/cropOwnDataset”中的结果，选择自己想要查询的行人，按照market1501的命名方式命名  
> 将同一个行人的不同照片重命名成“xxxx_xx”，其中前4位是行人ID，后2位是该照片ID，例：第1个行人的第2张照片：0001_02  
> 将制作好的行人底库图片放在“项目所在目录/data/querySet”中


----------------------------------------------------
## 6 测试

6.1 获取om模型
```
步骤详见4： 模型转换
```
6.2 准备数据集
```
步骤详见5： 数据集
```
6.3 配置环境变量
```   
#执行如下命令，打开.bashrc文件
cd $home
vi .bashrc
#在.bashrc文件中添加以下环境变量:

export MX_SDK_HOME=${SDK安装路径}/mxVision

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/

export PYTHONPATH=${MX_SDK_HOME}/python

export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner

export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

#保存退出.bashrc
#执行如下命令使环境变量生效
source ~/.bashrc

#查看环境变量
env
```
6.4 配置pipeline  
根据所需场景，配置pipeline文件，调整路径参数等。
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
    # 配置mxpi_tensorinfer插件的ReID.om模型加载路径（lines 53-60 以及 92-99）
    lines 53-60：
        "mxpi_tensorinfer1": {
            "props": {
                "dataSource": "mxpi_imagecrop0",
                "modelPath": "models/ReID.om(这里根据你的命名或路径进行更改)"
            },
            "factory": "mxpi_tensorinfer",
            "next": "appsink0"
        },
    lines 92-99：
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "models/ReID.om(这里根据你的命名或路径进行更改)"
            },
            "factory": "mxpi_tensorinfer",
            "next": "appsink0"
        }, 

```
6.5 执行
```
python3.7 main.py --queryFilePath='data/querySet' --galleryFilePath='data/gallerySet' --matchThreshold=0.3
```
> matchThreshold是行人重定位的阈值，默认值是0.3，可根据行人底库的数量进行调整，建议的范围是0.2~0.4之间  
> 如果使用自制数据集，可能由于数据噪声问题导致误报或漏检，此时将阈值酌情调大可减缓  
> 尽可能选择背景与行人区别较为明显的图片作为自制数据集

6.6 查看结果  
执行`main.py`文件后，可在“项目所在目录/result”路径下查看结果。


## 7 参考链接
> 特定行人检索：[Person Search Demo](https://github.com/songwsx/person_search_demo)  
