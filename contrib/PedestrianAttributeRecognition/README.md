# MindX SDK-行人属性检测

## 1 介绍
本开发样例是基于MindX SDK开发的端到端的Python应用实例，可在昇腾芯片上进行行人属性识别，并把可视化结果保存到本地。开发端到端行人属性识别，实现对检测图片中行人的定位与属性识别，并达到精度要求。该Sample的主要处理流程为：数据输入>预处理>行人检测>抠图缩放>行人属性识别>结果可视化

### 1.1 支持的产品

支持昇腾310芯片推理。

### 1.2 支持的版本

支持的SDK版本为2.0.2.b011。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 软件方案介绍

基于MindX SDK的行人属性识别业务流程：待检测图片通过appsrc插件输入，然后使用图像解码插件mxpi_imagedecoder对图片进行解码，再通过图像缩放插件mxpi_imageresize将图像缩放至满足行人检测模型(yolov3)要求的输入图像大小要求，缩放后的图像输入模型推理插件mxpi_tensorinfer中进行推理，随后将数据送入后处理插件mxpi_objectpostprocessor中进行处理，将得到的结果经过分发插件mxpi_distributor输出，随后将数据输出到裁剪插件mxpi_imagecrop中，根据上游推理插件推理出的结果进行裁剪出行人，并将裁剪后的图像输入到Deepmar的模型推理插件中，进行行人属性推理，最后将行人属性推理插件经序列化插件mxpi_dataserialize输出，即得到属性的预测结果，并将结果进行标签化，即为该图片的属性推理结果。

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统             | 功能描述                                                     |
| ---- | ------------------ | ------------------------------------------------------------ |
| 1    | 图片输入           | 获取jpg格式的输入图片                                        |
| 2    | 图片解码           | 解码图片                                                     |
| 3    | 图片缩放           | 将输入图片缩放到模型指定输入的尺寸大小                       |
| 4    | 行人检测           | 将输入的张量送入Yolov3模型推理插件中进行推理，得到行人检测框 |
| 5    | Yolov3模型后处理   | 对Yolov3推理插件的输出结果进行处理                           |
| 6    | Yolov3检测结果分发 | 对检测到的person类进行分发                                   |
| 7    | 图像裁剪           | 对Yolov3检测出的行人进行裁剪                                 |
| 8    | 行人属性检测       | 将裁剪后的图像数据送入Deepmar推理插件中进行属性推理          |
| 9    | 结果序列化         | 将35个属性推理结果序列化输出                                 |

### 1.4 代码目录结构与说明

本sample工程名称为yolov3_deepmar，工程目录如下图所示：
```
.
├── dataset
│   ├── image
│   ├── test_image
│   ├── image_jpg
│   ├── PETA.mat
│   ├── png2jpg.py
│   ├── transform_peta.py
│   ├── peta_dataset.pkl
│   └── peta_partition.pkl
├── model
│   ├── deepmar
│       ├── deepmar_bs1_aipp_1.om
│       ├── deepmar_bs1_unaipp.om
│   ├── yolov3
│       ├── coco.names
│       ├── yolov3_tf_bs1_fp16.cfg
│       ├── yolov3_tf_bs_fp16.om
├── pipeline
│   ├── test.pipeline
│   ├── test_only_deepmar.pipeline
├── evaluate.py
├── evaluate_for_deepmar.py
├── License
├── simgei.ttf
├── main.py
├── README.md
└── run.sh
```



## 2 环境依赖

推荐系统为ubantu 18.04，环境依赖软件和版本如下表：

| 软件名称    | 版本   |
| ----------- | ------ |
| python      | 3.7.5  |
| mxVision    | 2.0.2  |
| pillow      | 8.0.1  |
| pickle5     | 0.0.11 |

确保环境中正确安装mxVision SDK。

模型转换所需ATC工具环境搭建参考链接：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0004.html

在编译运行项目前，需要设置环境变量：

- 环境变量介绍
将${SDK安装路径}替换为自己的SDK安装路径; 
```python
export MX_SDK_HOME= ${SDK安装路径}/mxVision
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

## 3 模型转换

本项目中用到的模型有：yolov3，deeomar两个模型。

yolov3的模型转换及下载参考华为昇腾社区[ModelZoo](https://www.hiascend.com/zh/software/modelzoo/detail/C/210261e64adc42d2b3d84c447844e4c7)。

deepmar离线模型的转换及下载参考华为昇腾社区[ModelZoo](https://www.hiascend.com/zh/software/modelzoo/detail/1/4c787d576d284d1fa482cfa0ec3d4fb7)，对于无aipp设置的离线模型的转换，只需将atc转换时的 --insert_op_conf参数删除即可。

对于deepmar离线模型的aipp的设置，如下:

```python
aipp_op{
    aipp_mode:static
    input_format : YUV420SP_U8

    src_image_size_w : 224
    src_image_size_h : 224

    crop: false
    load_start_pos_h : 0
    load_start_pos_w : 0
    crop_size_w : 224
    crop_size_h: 224

    csc_switch : true
    rbuv_swap_switch : false

    mean_chn_0 : 123
    mean_chn_1 : 116
    mean_chn_2 : 103
    min_chn_0 : 0.675
    min_chn_1 : 0.28
    min_chn_2 : 0.53
    var_reci_chn_0: 0.0171
    var_reci_chn_1: 0.0175
    var_reci_chn_2: 0.0174

    matrix_r0c0: 256
    matrix_r0c1: 0
    matrix_r0c2: 359
    matrix_r1c0: 256
    matrix_r1c1: -88
    matrix_r1c2: -183
    matrix_r2c0: 256
    matrix_r2c1: 454
    matrix_r2c2: 0
    input_bias_0: 16
    input_bias_1: 128
    input_bias_2: 128}
```


## 3 编译与运行

**步骤1** 下载yolov3的离线模型：[yolo_tf_bs1_fp16.om](https://www.hiascend.com/zh/software/modelzoo/detail/C/210261e64adc42d2b3d84c447844e4c7),点击yolov3离线模型链接，选择“历史版本”中的版本1.1下载，将“yolov3_framework_tensorflow_ascend310_input_fp32_batch_1_fp16_output_fp32.om
”更名为：“yolo_tf_bs1_fp16.om”，并将其拷贝到本样例model文件夹下的yolov3文件夹中

**步骤2** 下载Deepmar的离线模型：[deepmar_bs1_aipp_1.om](https://www.hiascend.com/zh/software/modelzoo/detail/1/4c787d576d284d1fa482cfa0ec3d4fb7),打开链接，点击“下载模型脚本”，将其下载到本地，并将其中的deepmar_bs1.om文件拷贝到本样例model文件夹中的deepmar文件夹中，并将其更名为“deepmar_bs1_aipp_1.om”

**步骤3** 将需要进行推理的行人图片放入/dataset/test_image文件夹下，并修改yolov3_deepmar.py中“img_path"为需要推理的图片路径，将test.pipeline中的后处理so文件修改问SDK安装路径。

**步骤4** 执行命令：python main.py,得到final_result.jpg可视化结果



## 4 精度测试

下载开源数据集Peta，[下载地址：密码：5vep](https://pan.baidu.com/share/init?surl=q8nsydT7xkDjZJOxvPcoEw)，

（1）将下载好的文件夹中”./dataset/peta/images/*.png“中的images文件夹放入样例代码中的dataset/image文件夹下；将PETA.mat文件放在样例代码的dataset文件夹下。

（2）将Peta数据集中的png格式图片转为jpg格式图片

​	修改png2img.py脚本中的“filePath”为Peta数据集中的png格式图片的路径，并运行下面命令：

```python
python png2jpg.py
```

（3）拆分数据集

修改transform_peta.py脚本中的petaPath为PETA.mat文件的路径，添加save_file参数的文件保存路径以及traintest_split_file的参数路径，即将"样例工程路径"修改为自己的代码运行路径并运行下面命令：

```python
python dataset/transform_peta.py 
```

（4）精度测试

修改下面代码：

```python
    for idx in partition['test'][0]:
        image.append(dataset['test_image'][idx])
        label_tmp = np.array(dataset['att'][idx])[dataset['selected_attribute']].tolist()
        label.append(label_tmp)
```

依次修改上面代码中"partition\['test'][0]"为"partition\['test'][0]"、"partition\['test'][1]"、"partition\['test'][2]"、"partition\['test'][3]"、"partition\['test'][4]"

并运行下面命令：

```python
python evaluate_for_deepmar.py
```

可得到五组测试结果，并将五组测试结果求平均，可得到最终的peta测试集的平均属性识别准确率。

## 5 常见问题

本样例是针对单个行人的图片进行推理的，所以在选择测试图片时，请选择只有一个行人的测试图片，其次，测试样例的像素大小要做在32\*32～8192\*8192之间

## 6 适用场景
本样例适用于常见行人所处场景，如街道、红路灯路口、商场等，对于所推理的图片中的行人对象要求尽可能的完整、清晰、无遮挡、无重叠等。
