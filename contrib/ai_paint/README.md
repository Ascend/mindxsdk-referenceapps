# CGAN Ai Painting

## 1 简介
  本开发样例基于MindX SDK实现了从结构化描述生成对应风景照片的功能。
  参考以下链接说明：
  - https://www.hiascend.com/zh/developer/mindx-sdk/landscape?fromPage=1
  - https://gitee.com/ascend/samples/tree/master/cplusplus/contrib/AI_painting


### 1.1 支持的产品

本项目以昇腾Atlas 500 A2为主要的硬件平台。
  
## 2 依赖

|软件名称    | 版本     |
|-----------|----------|
| python    | 3.9.2    |
| MindX SDK |  5.0.rc1   |
| CANN | 310使用6.3.RC1<br>310B使用6.2.RC1 |

本样例无需外部依赖，在安装完成的SDK运行环境中即可执行。


## 3 环境变量

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh #toolkit默认安装路径，根据实际安装路径修改
. ${SDK_INSTALL_PATH}/mxVision/set_env.sh
```

## 4 模型转换
  获取原模型PB模型，通过atc工具可转换为对应OM模型。  

  - https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/painting/AIPainting_v2.pb

转换命令

```
atc --output_type=FP32 --input_shape="objs:9;coarse_layout:1,256,256,17"  --input_format=NHWC --output="AIPainting_v2" --soc_version=Ascend310B1 --framework=3  --model="AIPainting_v2.pb"
```


## 5 目录结构

```
.
|-------- model
|--------   |---- AIPainting_v2.om         //转换后的OM模型
|--------   |---- AIPainting_v2.pb         //原始PB模型
|-------- pipeline
|           |---- ai_paint.pipeline        //流水线配置文件          
|-------- python
|           |---- main.py                      //测试样例
|           |---- net_config.ini               //模型输入参数与说明
|           |---- run.sh                       //样例运行脚本
|-------- result
|-------- README.md 
```

## 6 运行

1. 获取om模型
2. 执行样例：
进入python目录， 修改net_config.ini文件中对应的网络参数，随后执行
```
python3 main.py
```

默认输出的矢量图layoutMap.jpg和结果图像resultImg.jpg位于result目录下

3. 性能测试：默认已包含单次生成的时间计算并在命令行输出
```shell
Time cost = 'xxx'ms
```