# CGAN Ai Painting

## 1 简介
  本开发样例基于MindX SDK实现了从结构化描述生成对应风景照片的功能。
  参考以下链接说明：
  - https://www.hiascend.com/zh/developer/mindx-sdk/landscape?fromPage=1
  - https://gitee.com/ascend/samples/tree/master/cplusplus/contrib/AI_painting

## 2 模型转换
  原模型为PB模型，通过atc工具可转换为对应OM模型。  
  本样例中在model目录下已提供om模型和对应转换脚本，如仓库限制无法获取大文件，请使用以下下载地址：
  - https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/painting/AIPainting_v2.pb
  - https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/painting/AIPainting_v2.om


## 3 目录结构

```
.
|-------- model
|--------   |---- AIPainting_v2.om         //转换后的OM模型
|--------   |---- AIPainting_v2.pb         //原始PB模型
|--------   |---- model_conversion.sh      //模型转换脚本
|-------- pipeline
|           |---- ai_paint.pipeline        //流水线配置文件          
|-------- python
|           |---- main.py                      //测试样例
|           |---- net_config.ini               //模型输入参数与说明
|           |---- run.sh                       //样例运行脚本
|-------- result
|-------- README.md 
```

## 4 依赖

|软件名称    | 版本     |
|-----------|----------|
| python    | 3.9.2    |
| MindX SDK | 2.0.4    |
| CANN | 5.0.4 |

本样例无需外部依赖，在安装完成的SDK运行环境中即可执行。

## 5 运行

1. 获取om模型
2. run.sh脚本中LD_LIBRARY_PATH设置了ACL动态库链接路径为/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64，如果实际环境中路径不一致，需要替换为实际的目录。
3. 如果环境变量中没有MX_SDK_HOME变量，则需要在run.sh脚本中设置MX_SDK_HOME变量为你实际的MX_SDK安装目录。默认配置中MX_SDK_HOME为样例位于SDK自带sample目录时的相对路径。
4. 若要执行样例：
修改python目录下net_config.ini文件中对应的网络参数，随后执行
```bash
bash run.sh
```
默认输出的矢量图layoutMap.jpg和结果图像resultImg.jpg位于result目录下
5. 性能测试：默认已包含单次生成的时间计算并在命令行输出
```shell
Time cost = 'xxx'ms
```