#  基于Mind SDK 的驾驶员状态识别

## 介绍

本开发样例是基于mxBase开发的端到端推理的应用程序，可在昇腾芯片上识别视频中的驾驶员状态，然后送给分类模型进行推理，再由模型后处理得到驾驶员的状态识别结果。 其中包含Rcf模型的后处理模块开发。 主要处理流程为：  输入视频>视频解码  >图像前处理 >分类模型推理 >分类模型后处理 >驾驶员状态

## 基于MindSpore框架训练模型

**步骤1** 训练数据获取 下载数据集 。[下载地址](https://www.kaggle.com/c/state-farm-distracted-driver-detection/leaderboard)

**步骤2** 将数据集按照 4：1 分为训练集和验证集

**步骤3** 训练代码下载  将获取到基于MindSpore的ResNet50模型 [下载地址](https://www.hiascend.com/zh/software/modelzoo/detail/C/ea8c34895d1b4697b3f1e940da1e97d2)

**步骤3** 训练模型

(1)、按照如下修改 train.py中的 init_loss_scale()

```
def init_loss_scale():
    if config.dataset == "imagenet2012" or  config.dataset == "distracted_driver_detection":  
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    return loss


```

(2）编译与运行

```
 cd scripts
 
 bash run_standalone_train.sh /path/to/distracted_driver_detection/train /path/to/resnet50_distracted_driver_detection_Acc_config.yaml
```

(3) 转换模型

```
python3 export.py --network_dataset resnet50_dirver_detection  --ckpt_file scripts/train/output/checkpoint/best_acc.ckpt  --file_name resnet50-dirver_detection-915-air --file_format AIR --config_path /path/to/resnet50_distracted_driver_detection_Acc_config.yaml
```

## SDK 运行

**步骤1** 设置环境变量

```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

**步骤2** 模型转换

```
 cd convert
 bash air2om.sh /path/to/*.air  /path/to/output_name_use_to_video_test.om  yuv_aipp.config
```

**步骤3**  启动rtsp服务

按照 [教程](https://gitee.com/ascend/mindxsdk-referenceapps/wikis/MindX SDK 指引页?sort_id=4015504) 自行准备数据 并启动rtsp服务

**步骤4** 修改配置文件

修改pipeline中的 "rtspUrl", "modelPath", "postProcessLibPath" 等选项

**步骤4** 进行驾驶员状态识别

```
bash run.sh main.py 30    
```

参数说明：

30： 检测时间段为30s

## 精度测试

(1) 模型转换

```
cd convert
bash mindir2om.sh /path/to/*.air  /path/to/output_name_use_to_percision_test.om  aipp.config
```

(2)  修改pipeline/dirver-detection-img.pipeline 中的相关选项，如om模型路径等

(3) 测试精度

```
bash run.sh percision.py /path/to/val_data  
```

## 性能测试

参考 "sdk 运行"章节准备数据、启动rtsp服务等步骤

```
bash run.sh  performance.py 
```
