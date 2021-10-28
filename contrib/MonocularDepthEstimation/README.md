# 基于MxStream的AdaBins单目深度估计

## 1 介绍

基于AdaBins室内模型的单目深度估计，输出输入图像的深度图 \
样例输入：室内图片（如果要得到输出大小和输入完全一致的深度图，请输入宽 320 * n，高 240 * n 的图片）\
样例输出：输入图片的深度图（灰度图形式）

### 1.1 支持的产品
昇腾310(推理)

### 1.2 支持的版本
本样例配套的CANN版本为 [3.3.0](https://www.hiascend.com/software/cann/commercial) ，MindX SDK版本为 [2.0.2](https://www.hiascend.com/software/mindx-sdk/mxvision) 。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 目录结构
```
.
|-------- depth_estimation
|           |---- monocular_depth_estimation.py     // 基于MxStream和Adabins模型的深度估计实现
|-------- image
|           |---- test.jpg                          // 测试的室内图片(需自行准备)
|-------- model
|           |---- aipp_adabins_640_480.aippconfig   // Adabins 模型转换配置文件
|           |---- model_conversion.sh               // 模型转换脚本
|-------- pipeline
|           |---- depth_estimation.pipeline         // 深度估计的pipeline文件
|-------- result                                    // 深度图结果或模型精度结果存放处
|-------- test_set
|           |---- image                             // 测试集图片(需自行准备)
|           |---- depth_info                        // 测试集图片深度信息(需自行准备)
|-------- util
|           |---- data_process.py                   // 处理测试集数据
|           |---- util.py                           // 公共方法
|-------- evaluate.py                               // 模型精度验证
|-------- main.py                                   // 单目深度估计样例
|-------- README.md                                 // ReadMe
|-------- run.sh                                    // 样例运行脚本

```
## 2 环境依赖

### 2.1 软件版本
| 软件                 | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 2.0.2       | mxVision软件包                  | [链接](https://www.hiascend.com/software/mindx-sdk/mxvision) |
| Ascend-CANN-toolkit | 3.3.0       | Ascend-cann-toolkit开发套件包    | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统             | Ubuntu 18.04 | 操作系统                        | Ubuntu官网获取                                               |


### 2.2 准备工作

> 模型转换

**步骤1** 在 [GitHub AdaBins](https://github.com/shariqfarooq123/AdaBins) 上下载预训练模型 [AdaBins_nyu.pt](https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA) ，
或者直接下载已经转换好的 [AdaBins_nyu.onnx](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MonocularDepthEstimation/AdaBins_nyu.onnx) ，跳过 **步骤2** 直接进入om模型的转换

**步骤2** 将获取到的 `AdaBins_nyu.pt` 转换为 `AdaBins_nyu.onnx`, [参考链接](https://blog.csdn.net/ApathyT/article/details/120834163)

**步骤3** 将转换或下载得到的 `AdaBins_nyu.onnx` 放在 `model` 目录下

**步骤4** 运行模型转换脚本 `model_conversion.sh` 或在 `model` 目录下执行以下命令

```
# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 执行，转换AdaBins模型
# Execute, transform AdaBins model.

atc --model=./AdaBins_nyu.onnx --framework=5 --output=./AdaBins_nyu.om --soc_version=Ascend310 --insert_op_conf=./aipp_adabins_640_480.aippconfig --log=error
```

执行完模型转换脚本后，会生成相应的.om模型文件。

模型转换使用了ATC工具，如需更多信息请参考:

 https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html

> 相关参数修改

**通用配置**
1) [depth_estimation.pipeline](./pipeline/depth_estimation.pipeline) 中配置 `AdaBin_nyu.om` 模型路径
```
"mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "${ AdaBin_nyu.om 模型路径}"
            },
            "factory": "mxpi_tensorinfer",
            "next": "appsink0"
        }
```
2) [monocular_depth_estimation.py](./depth_estimation/monocular_depth_estimation.py) 中配置模型的实际输出宽高
```python
# depth estimation model output size
model_output_height = 240
model_output_width = 320
```

**推理配置**
1) 准备一张室内图片，置于 [image](./image) 文件夹中
2) [main.py](./main.py) 中配置 `input_image_path` 和 `output_result_path`
```
    input_image_path = 'image/${测试图片文件名}'   # 仅支持jpg格式
    output_result_path = "result/${输出结果文件名}" 
```

**精度测试配置**
1) 下载测试集数据（ [下载地址](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) ）, 
   置于 [test_set](./test_set) 文件夹下，检查文件名是否为 `nyu_depth_v2_labeled.mat`
2) 运行 [data_process.py](./util/data_process.py)，处理完毕后的图片位于 [test_set/image](./test_set/image) 文件夹下，
   深度信息位于 [test_set/depth_info](./test_set/depth_info) 文件夹下 
3) [evaluate.py](./evaluate.py) 中配置测试集图片大小
```
   # test set image size
   test_image_height = ${测试集图片的高}
   test_image_width = ${测试集图片的宽}
```
4) [evaluate.py](./evaluate.py) 其他可选配置项
```
   # thresholds for accuracy
   threshold_1 = 1.25
   threshold_2 = 1.25 ** 2
   threshold_3 = 1.25 ** 3
```

### 2.3 配置环境变量

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

## 3 运行
手动运行请参照 ①， 脚本运行请参照 ②
> ① 手动运行前请确保每一步环境变量均配置完成，随后进入工程目录，键入执行指令
```bash
# 进入工程目录
cd MonocularDepthEstimation

# 图片深度估计
python3.7 main.py ${测试图片路径} ${输出结果路径}
ex: python3.7 main.py image/test.jpg result/result.jpg

# AdaBins_nyu 模型精度验证
python3.7 evaluate.py
```

> ② 脚本运行请先赋予可执行权限
```bash
# 赋予可执行权限
chmod +x run.sh

# 说明：-m 运行模式 {infer | evaluate} -i infer模式下输入图片的路径 -o infer模式下输出结果的路径
# 推理模式
bash run.sh -m infer -i image/test.jpg -o result/result.jpg
# 精度验证模式
bash run.sh -m evaluate
```

## 4 查看结果
执行`run.sh`完毕后，sample会将**图片深度信息**或**模型精度结果**保存在工程目录下`result`中。