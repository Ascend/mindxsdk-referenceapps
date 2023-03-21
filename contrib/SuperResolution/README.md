# 基于MxVision的VDSR图像超分辨率

## 介绍

基于MindX SDK mxVision开发图像超分辨率程序。本程序采用python开发，通过预处理操作对输入的图片数据解码为YUV格式图片，并将解码后的图片缩放到模型推理要求的尺寸。然后利用图像超分辨率模型VDSR获取得到图片超分辨率重建结果。最后，利用python的第三方图像处理库PIL将低分辨率的输入图像和超分辨率重建的结果一同可视化。其次，针对两个图片集91-images和General-100进行PSNR（峰值信噪比）验证。

程序输入：任意jpg图片
程序输出：输出得到低分辨率图片（256 x 256px）和超分辨率重建图片（768 x 768px）组合的可视化大图

### 支持的产品

昇腾310B(推理)

## 环境依赖 

### 软件版本

| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 5.0 RC1        | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| Ascend-CANN-toolkit | 6.2       | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统            | Ubuntu 18.04 | 操作系统                      | Ubuntu官网获取    

### 目录结构

```
.
|
|-------- font
|           |---- SourceHanSansCN-Normal-2.otf      // otf字体
|-------- image
|           |---- test.jpg                          // 测试图片(需自行准备)
|-------- result                                    // 测试图片程序输出存放处
|-------- model
|           |---- YUV420SP_U8_GRAY.cfg              // 模型转换配置文件(灰度图)
|           |---- model_conversion.sh               // 模型转换脚本
|           |---- VDSR_768_768.om                   // 转换后OM模型存放在此处(需自行上传)
|-------- testSet
|           |---- 91-images                         // 91-images验证集(含bmp图片)
|           |---- 91-images-jpg                     // 91-images验证集转换后集(含jpg图片)
|           |---- output                            // 验证集结果输出目录
|           |---- bmp2jpg.py                        // bmp转jpg脚本
|-------- evaluate.py                               // 模型精度验证
|-------- README.md                                 // ReadMe
|-------- main.py                                   // 图像超分辨率主程序
|-------- util.py                                   // 工具方法

```
                                           

### 准备工作

> 配置环境变量

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh #toolkit默认安装路径，根据实际安装路径修改
. ${SDK_INSTALL_PATH}/mxVision/set_env.sh

```

> 模型转换

**步骤1** 获取原始模型网络及权重, [权重下载 和 网络下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/SuperResolution/model.zip)

**步骤2** AIPP配置文件-YUV420SP_U8_GRAY.cfg

```cfg
aipp_op {
    aipp_mode: static
    input_format : YUV420SP_U8
    csc_switch : true
    rbuv_swap_switch : false
    matrix_r0c0 : 256
    matrix_r0c1 : 0
    matrix_r0c2 : 0
    matrix_r1c0 : 0
    matrix_r1c1 : 0
    matrix_r1c2 : 0
    matrix_r2c0 : 0
    matrix_r2c1 : 0
    matrix_r2c2 : 0
    input_bias_0 : 0
    input_bias_1 : 0
    input_bias_2 : 0
}
```

色域转换，用于将输入的图片格式，转换为模型需要的图片格式，这里将YUV420SP_U8转GRAY，详细色域转换配置说明请参考 [这里](https://support.huawei.com/enterprise/zh/doc/EDOC1100191944/3e66a3c#ZH-CN_TOPIC_0000001095142890)

**步骤3** 将下载得到模型网络及权重(`VDSR.prototxt`、`VDSR.caffemodel`)、AIPP配置文件(`YUV420SP_U8_GRAY.cfg`)放在 `model` 目录下

**步骤4** 使用ATC模型转换工具进行模型转换

`model` 目录下执行以下命令

```
# 执行，转换VDSR模型成om格式
# Execute, transform VDSR model.

atc --model=./VDSR.prototxt --weight=./VDSR.caffemodel --framework=0 --input_format=NCHW --input_shape="data: 1, 1, 768, 768" --output=./VDSR_768_768 --soc_version=Ascend310B1 --output_type=FP32 --insert_op_conf=YUV420SP_U8_GRAY.cfg
```

执行完模型转换脚本后，会在model目录下生成相应的VDSR_768_768.om模型文件。

模型转换使用了ATC工具，如需更多信息请参考 [这里](https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md)

> pipeline中模型路径

在`main.py`中配置 `VDSR_768_768.om` 模型路径

```
"mxpi_tensorinfer0": {
    "props": {
        "dataSource": "mxpi_imagedecoder0",
        "modelPath": "model/VDSR_768_768.om"
    },
    "factory": "mxpi_tensorinfer",
    "next": "appsink0"
}
```

## 运行

1) 准备一张测试图片，置于 image 文件夹中（仅支持jpg格式）

2) 进入工程目录，键入执行指令，发起推理性能测试：

```python
python3 main.py ${测试图片路径}
例如: python3 main.py image/head.jpg
    
# 或者在main.py中配置 input_image_path 
input_image_path = 'image/${测试图片文件名}'   # 仅支持jpg格式
```

执行完毕后，sample会将程序输出的可视化结果，保存在工程目录下`result`中 (可视化结果名和输入图片一致)



## 精度验证

PSNR（峰值信噪比）经常用作图像压缩等领域中信号重建质量的测量方法。

1）准备测试集：下载验证图片集，[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/SuperResolution/testImageSet.zip)，从zip文件中取出两个图片集91-images和General-100放置到testSet目录下，其中91-images包含91张t\*.bmp图片，General-100包含100张im_\*.bmp图片

2）图片格式转换：参考`testSet/bmp2jpg.py`脚本，将两个图片集中bmp图片转换为jpg图片

进入`testSet`目录，键入执行指令，图片格式转换：

```python
python3 bmp2jpg.py ${测试图片集路径}
例如: python3 bmp2jpg.py 91-images
    
# 或者在evaluate.py中配置 test_image_set_path 
test_image_set_path = './${测试图片集路径}'
```

然后会在`testSet`目录下，生成转换后的包含jpg格式图片的文件夹，文件夹名称为`${测试图片集路径}-jpg`

3）利用`evaluate.py `脚本，计算得到两个图片集的平均PSNR（峰值信噪比）

键入执行指令，发起精度验证测试：

```python
python3 evaluate.py ${验证图片集路径}
例如: python3 evaluate.py testSet/91-images-jpg
    
# 或者在evaluate.py中配置 test_image_set_path 
test_image_set_path = './${验证图片集路径}'
```

执行完毕后，会在控制台输出该验证图片集的平均峰值信噪比