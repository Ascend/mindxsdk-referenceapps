# 视频伪装物体检测

## 1 介绍

基于 MindX SDK 实现 SLT-Net 模型的推理，在 MoCA-Mask 数据集上 Sm 达到大于 0.6。输入连续几帧伪装物体的视频序列，输出伪装物体掩膜 Mask 图。


### 1.1 支持的产品

支持昇腾310芯片


### 1.2 支持的版本

支持的SDK版本：SDK3.0 RC2

版本号查询方法，在Atlas产品环境下，运行命令：

```
npu-smi info
```


### 1.3 软件方案介绍

本方案中，`torch2onnx.py` 将 PyTorch 版本的伪装视频物体检测模型 [SLT-Net](https://github.com/XuelianCheng/SLT-Net)，转化为晟腾的om模型。`inference.py` 将输入视频帧进行处理，通过调用晟腾om模型进行处理，生成最终的视频伪装物体的掩膜 Mask 图。


### 1.4 代码目录结构与说明

本sample工程名称为 VCOD_SLTNet，工程目录如下图所示：

```
──VCOD_SLTNet
    ├── flowchart.jpeg
    ├── inference.py   # 推理文件
    ├── torch2onnx.py  # 模型转换脚本
    └── README.md
```


### 1.5 技术实现流程图

![Flowchart](./flowchart.jpeg)

图1 视频伪装物体检测流程图


### 1.6 特性及适用场景

对于伪装视频数据的分割任务均适用，输入视频需要转换为图片序列输入到模型中，具体可以参考 MoCA 数据格式，[MoCA 测试样例（华为 obs 链接）](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/sltnet/MoCA_Video.zip) ，分辨率在 (352*352) 左右。


## 2 环境依赖

环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| -------- | ------ |
| MindX SDK  | mxVision-3.0.RC2 |
| Python |   3.9.2     |
|  CANN        |  5.1RC2    |
| PyTorch | 1.12.1 |
| numpy | 1.21.5 |
| imageio | 2.22.3| 
| Pillow | 9.3.0 | 
| cv2 | 4.5.5 |
| timm | 0.4.12 |
| tqdm | 4.64.1 |

在编译运行项目前，需要设置环境变量：

- 环境变量介绍

```
MindSDK 环境变量：
    . ${SDK-path}/set_env.sh
CANN 环境变量：
    . ${ascend-toolkit-path}/set_env.sh
环境变量介绍
SDK-path: mxVision SDK 安装路径
ascend-toolkit-path: CANN 安装路径。
```


## 3. 数据准备


### 3.1 下载相关文件

- 运行 inference 下载 om 模型文件：[华为 obs 链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/sltnet/models.zip)，其中也包含 om 文件生成用到的 torch 模型文件与 onnx 文件。

- 运行 torch2onnx 下载 PyTorch 版本 [SLT-Net 代码](https://github.com/XuelianCheng/SLT-Net)，[SLT-Net 模型文件](https://drive.google.com/file/d/1_u4dEdxM4AKuuh6EcWHAlo8EtR7e8q5v/view) ，保留 `Net_epoch_MoCA_short_term_pseudo.pth` 文件即可，也可以通过 [华为 obs 链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/sltnet/models.zip) 来下载 torch 模型文件。然后进行下述操作来转换模型。[SLT_Net_MindXsdk_torch](https://github.com/shuowang-ai/SLT_Net_MindXsdk_torch) 为已经修改过的代码样例，供参考，并且，该链接提供精简的 [评测指标](https://github.com/shuowang-ai/SLT_Net_MindXsdk_torch/tree/master/eval_python) 的运行代码

- 下载数据集 [MoCA](https://drive.google.com/file/d/1FB24BGVrPOeUpmYbKZJYL5ermqUvBo_6/view) ，或者通过 [MoCA 测试样例（华为 obs 链接）](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/sltnet/MoCA_Video.zip) 来下载


### 3.2 修改 [SLT-Net 代码](https://github.com/XuelianCheng/SLT-Net) 以完成模型转换，使用 [SLT_Net_MindXsdk_torch](https://github.com/shuowang-ai/SLT_Net_MindXsdk_torch) 可跳过该步骤。

1. `lib/__init__.py` 中注释掉第二行

因为长期模型依赖 CUDA，并且需要在 CUDA 平台进行编译，而本项目基于 MindX SDK 实现，因此使用短期模型。并且，短期模型的评价指标满足预期。

```
from .short_term_model import VideoModel as VideoModel_pvtv2
# from .long_term_model import VideoModel as VideoModel_long_term
```

2. `lib/short_term_model.py`

```
image1, image2, image3 = x[0],x[1],x[2]
```

替换为

```
image1, image2, image3 = x[:, :3], x[:, 3:6], x[:, 6:]
```

3. `lib/pvtv2_afterTEM.py`

注释掉
```
# from mmseg.models import build_segmentor
# from mmcv import ConfigDict
```

4. `mypath.py`

```
elif dataset == 'MoCA':
    return './dataset/MoCA-Mask/'
```

替换为数据集路径 `TestDataset_per_sq` 的父目录。


## 4. 模型转换

### 4.1 设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/mindx_dir/mxVision/set_env.sh
conda activate py392
```

### 4.2 执行编译的步骤

1. pytorch 模型转换 onnx 文件


    将 `torch2onnx.py` 放到 SLT-Net 项目根目录。推荐下载已经完成针对模型转化优化的项目代码 [SLT_Net_MindXsdk_torch](https://github.com/shuowang-ai/SLT_Net_MindXsdk_torch)。
    
    下载 [模型文件](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/sltnet/models.zip)，指定其中的 torch 模型文件路径，如： `./sltnet.pth`

    注意，timm 的版本为 `0.4.12`，其他版本可能会报错。

    pytorch 模型转换 onnx 涉及到的最基本的环境如下：

    ```
    conda create -n vcod_sltnet python=3.8
    conda activate vcod_sltnet
    conda install pytorch==1.12.1 -c pytorch
    pip install timm==0.4.12
    ```

    运行：

```
python torch2onnx.py --pth_path ./sltnet.pth --onnx_path ./sltnet.onnx
```

2. 简化 onnx 文件 （可选操作）

```
python -m onnxsim --input-shape="1,9,352,352" --dynamic-input-shape sltnet.onnx sltnet_sim.onnx
```

3. onnx 文件转换 om 文件

```
atc --framework=5 --model=sltnet.onnx --output=sltnet --input_shape="image:1,9,352,352" --soc_version=Ascend310 --log=error
```

注意，该操作耗时较长（约1小时左右），如无报错，请耐心等待。

已经转换好的模型可供参考：[模型文件](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/sltnet/models.zip)


## 5. 运行推理

mindspore 版本模型：配置代码中参数：1. 输出结果保存目录 `save_root`； 2. om 模型路径 `om_path`，直接运行 inference.py 即可。无需放入 SLT-Net 根目录。

```
python inference.py --datapath ${MoCA_Video数据集路径} --save_root ./results/ --om_path ./sltnet.om --testsize 352 --device_id 0
```

参数说明：
    datapath：下载数据以后，目录中，`TestDataset_per_sq` 的上一级目录，
    save_root：结果保存路径
    om_path：om 模型路径
    testsize：图片 resize 的大小，当前固定为 352
    device_id：设备编号


## 6. 精度评估


- 可以使用基于 [MATLAB](https://github.com/XuelianCheng/SLT-Net/tree/master/eval) 或基于 [Python](https://github.com/lartpang/PySODEvalToolkit) 的评测代码。

- 推荐使用精简后的 [Python 评测代码](https://github.com/shuowang-ai/SLT_Net_MindXsdk_torch/tree/master/eval_python)

- 运行 `eval/run_eval.py` 脚本，修改 `gt_dir`、`pred_dir` 为本地的 GT、预测结果的目录即可。

- 运行

```
python eval/run_eval.py
```

得到指标结果

```
{'Smeasure': 0.6539, 'wFmeasure': 0.3245, 'MAE': 0.0161, 'adpEm': 0.6329, 'meanEm': 0.7229, 'maxEm': 0.7554, 'adpFm': 0.3025, 'meanFm': 0.3577, 'maxFm': 0.3738}
```
