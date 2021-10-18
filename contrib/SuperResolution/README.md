# 基于MxVision的FSRCNN图像超分辨率

## 介绍

基于MindX SDK 2.0.1 mxVision开发图像超分辨率程序。本程序采用python开发，通过预处理操作对输入的图片数据解码为YUV格式图片，并将解码后的图片缩放到模型推理要求的尺寸。然后利用图像超分辨率模型FSRCNN获取得到图片超分辨率重建结果。最后，利用python的第三方图像处理库PIL将低分辨率的输入图像和超分辨率重建的结果一同可视化。其次，针对两个图片集91-images和General-100进行PSNR（峰值信噪比）验证。

程序输入：任意jpg图片
程序输出：输出得到低分辨率图片（256 x 256px）和超分辨率重建图片（768 x 768px）组合的可视化大图

## 目录结构

```
super_resolution
|
|-------- font
|           |---- SourceHanSansCN-Normal-2.otf      // otf字体
|-------- image
|           |---- test.jpg                          // 测试图片(需自行准备)
|-------- result                                    // 测试图片程序输出存放处
|-------- model
|           |---- YUV420SP_U8_GRAY.cfg              // 模型转换配置文件(灰度图)
|           |---- model_conversion.sh               // 模型转换脚本
|           |---- FSRCNN_256_256.om                 // 转换后OM模型存放在此处
|-------- testSet
|           |---- 91-images-jpg                     // 91-images图片验证集
|           |---- general-100-jpg                   // general-100图片验证集
|           |---- output                            // 验证集结果输出目录
|           |---- evaluate.py                       // 模型精度验证
|-------- README.md                                 // ReadMe
|-------- main.py                                   // 图像超分辨率主程序
|-------- util.py                                   // 工具方法

```

## 准备工作

> 模型转换

**步骤1** 获取原始模型网络及权重, [权重下载地址](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/super_resolution/FSRCNN/FSRCNN.caffemodel)、[网络下载地址](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/super_resolution/FSRCNN/FSRCNN.prototxt)

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

**步骤3** 将下载得到模型网络及权重(`FSRCNN.prototxt`、`FSRCNN.caffemodel`)、AIPP配置文件(`YUV420SP_U8_GRAY.cfg`)放在 `model` 目录下

**步骤4** 使用ATC模型转换工具进行模型转换

设置环境变量（请确认install_path路径是否正确）

```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

参照指令执行，转换FSRCNN模型成om格式

```
atc --model=./FSRCNN.prototxt --weight=./FSRCNN.caffemodel --framework=0 --input_format=NCHW --input_shape="data: 1, 1, 256, 256" --output=./FSRCNN_256_256 --soc_version=Ascend310 --output_type=FP32 --insert_op_conf=YUV420SP_U8_GRAY.cfg
```

执行完模型转换脚本后，会在model目录下生成相应的FSRCNN_256_256.om模型文件。

模型转换使用了ATC工具，如需更多信息请参考 [这里]( https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html)


## 推理

1) 配置环境变量

```bash
# 执行如下命令，打开.bashrc文件
cd $HOME
vi .bashrc

# 在.bashrc文件中添加以下环境变量
MX_SDK_HOME=${SDK安装路径}
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

2) 准备一张室内图片，置于 image 文件夹中（仅支持jpg格式）

2) 进入工程目录，键入执行指令，发起推理性能测试：

```python
python3.7 main.py ${测试图片路径}
例如: python3.7 main.py image/head.jpg
    
# 或者在main.py中配置 input_image_path 
input_image_path = 'image/${测试图片文件名}'   # 仅支持jpg格式
```

执行完毕后，sample会将程序输出的可视化结果，保存在工程目录下`result`中 (可视化结果名和输入图片一致)



## 精度验证

PSNR（峰值信噪比）经常用作图像压缩等领域中信号重建质量的测量方法，它常简单地通过均方差（***\*MSE\****）进行定义

1. 下载验证图片集，[下载地址](https://drive.google.com/file/d/0B7tU5Pj1dfCMWjhhaE1HR3dqcGs/view?resourcekey=0-bp1OwS1HzuCuelcMG0sRvQ)，从zip文件中取出两个图片集91 images和General-100放置到testSet目录下，其中91 images包含91张t\*.bmp图片，General-100包含100张im_\*.bmp图片

2. 参考如下python脚本，两个图片集中bmp图片转换为jpg图片

   ```python 
   from PIL import Image
   if __name__ == '__main__':    
       for i in range(1, 92, 1):
           image_file = './91-images/t'+str(i)+'.bmp'
           out_img = Image.open(image_file).convert("RGB")
           out_img_name = 't_'+str(i)+'.jpg'
           out_img.save("./91-images-jpg/"+out_img_name)
   
       for i in range(1, 101, 1):
           image_file = './General-100/im_'+str(i)+'.bmp'
           out_img = Image.open(image_file).convert("RGB")
           out_img_name = 'im_'+str(i)+'.jpg'
           out_img.save("./general-100-jpg/"+out_img_name)
   ```

3. 利用`evaluate.py `脚本，计算得到两个图片集的平均PSNR（峰值信噪比）

4. 执行完毕后，低分辨率的输入图像和超分辨率重建的结果一同可视化图片，存放在testSet/output目录下

