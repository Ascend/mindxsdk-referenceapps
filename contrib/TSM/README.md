# TSM视频分类参考设计

## 1 介绍
使用TSM模型，基于Kinetics-400数据集，在MindX SDK环境下实现视频分类功能。将测试视频传入脚本进行前处理，模型推理，后处理等功能，最终得到模型推理的精度和性能。

### 1.1 支持的产品

以昇腾Atlas310卡为主要的硬件平台

### 1.2 支持的版本

CANN:3.0RC2

SDK:mxvision 5.1RC2(可通过cat SDK目录下的version.info查看)

eg：版本号查询方法，在Atlas产品环境下，运行命令：

```
npu-smi info
```

### 1.3 代码目录结构与说明

```text
├── TSM
    ├── README.md                        // 所有模型相关说明
    ├── ops
        ├── basic_ops.py
        ├── dataset.py                   // 数据集构建脚本
        ├── dataset_config.py            // 数据集配置脚本
        ├── models.py                    // 模型搭建脚本 
        ├── temporal_shift.py
        ├── transforms.py
    ├── model
        ├── pth2onnx.py                  // 转onnx模型脚本
        ├── onnx2om.sh                   // 转om模型脚本
        ├── mobilenet_v2_tsm.py          // 在线模型脚本
        ├── pth2onnx1.py                 // 在线模型转onnx模型脚本
        ├── onnx2om1.sh                  // 在线模型转om模型脚本
    ├── tools 
        ├── gen_label_kinetics.py        // label生成脚本
        ├── vid2img_kinetics.py          // 视频抽帧脚本
        ├── kinetics_label_map.txt
    ├── label
        ├── kinetics_val.csv             // label文件
    ├── download_data
        ├── download.sh                  // 下载数据集脚本
        ├── k400_extractor.sh            // 解压数据集脚本
        ├── val_link.list    
    ├── online_infer.py                  // 在线推理精度脚本
    ├── offline_infer.py                 // 离线推理精度脚本
    ├── speed.py                         // 离线推理NPU性能脚本
    ├── speedGPU.py                      // 离线推理GPU性能脚本
```

## 2 环境依赖

推荐系统为ubuntu 18.04，环境依赖软件和版本如下表

| 软件名称     | 版本     |
|----------|--------|
| cmake    | 3.5+   |
| mxVision | 5.1RC2 |
| Python   | 3.9    |
| torch    | 1.10.0 |
| ffmpeg   | 3.4.8  |

- 环境变量介绍

```Shell
export PYTHONPATH=${SDK路径}/MindX_SDK/mxVision-3.0.RC2/python:$PYTHONPATH
. /usr/local/Ascend/ascend-toolkit/set_env.sh
. ${SDK路径}/MindX_SDK/mxVision-3.0.RC2/set_env.sh
```

## 3  离线推理

**步骤1** Kinetics-400数据集下载

在“/TSM/download_data/”目录下，运行数据集下载脚本，在“/TSM/”目录下新建文件夹data用于存放下载的数据集

```Shell
bash download.sh val_link.list
bash k400_extractor.sh
```

**步骤2** 数据集预处理

1、视频抽帧

在“/TSM/tools/”目录下，运行视频抽帧脚本，在“/TSM/”目录下新建文件夹dataset用于存放视频抽帧后的图片

```shell
python3 vid2img_kinetics.py [video_path] [image_path]
e.g.
python3 vid2img_kinetics.py ../data ../dataset/
```

在“/TSM/tools/”目录下，运行脚本生成标签

```shell
python3 gen_label_kinetics.py
```

修改参数

```python
import os


dataset_path = '../dataset'           # 放视频抽帧后的图片路径
label_path = '../label'               # 存放label路径
```

**步骤3** 模型转换

下载模型，模型连接 https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/TSM/online_models.zip
将下载好的模型放在“/TSM/model”目录下。

将模型转换为onnx模型，在“/TSM/model/”目录下，运行脚本生成onnx模型

```shell
python3 pth2onnx.py  kinetics  RGB
```

将模型转换为om模型，在“/TSM/model/”目录下，运行脚本生成om模型

```shell
bash onnx2om.sh
```

**步骤4** 精度测试

在“/TSM/”目录下，运行精度测试脚本

```shell
python3 offline_infer.py kinetics --test_segments=8 --test_crops=1 --batch_size=1 --full_res
```
**步骤5** 性能测试

将用来测试的单视频放在“/TSM/”目录下，如视频“test.mp4”，运行性能测试脚本

修改参数，'./test.mp4'为测试视频

```python
def main():
    cmd = 'ffmpeg  -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 0 \"{}/img_%05d.jpg\"'.format('./test.mp4', './image')
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    files = os.listdir(r"./image/")
```

GPU性能

```shell
python3 speedGPU.py kinetics --test_segments=8 --test_crops=1 --batch_size=1
```

SDK性能

```shell
python3 speed.py
```

## 4 在线推理

**步骤1** 安装视频流工具

参考 https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/reference_material/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md

1、下载安装包

https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/tool/live555.tar.gz

2、解压

```shell
tar -zxvf live555.tar.gz

tar -zxvf live.tar.gz

解压后目录
|--live
|--live.tar.gz
|--live555.tar.gz
|--startNvr.sh
```

3、编译并安装

```shell
cd live/
./genMakefiles linux  #注意后面这个参数是根据当前文件夹下config.<后缀>获取得到的,与服务器架构等有关。

make
```

最后就会在mediaServer 目录下生成一个live555MediaServer可执行文件

**步骤2** 生成视频流

使用ffmpeg工具将mp4格式视频生成264格式视频

```shell
ffmpeg -i test.mp4 -vcodec h264 -bf 0 -g 25 -r 10 -s 1280*720 -an -f h264 test.264

//-bf B帧数目控制，-g 关键帧间隔控制，-s 分辨率控制 -an关闭音频， -r 指定帧率
```

生成视频流

```shell
mkdir ${your_video_path}
cd ${your_video_path}
cp ${解压路径}/startNvr.sh ./
cp ${h264视频} ./
cp ${live_path}/mediaServer/live555MediaServer ./

chmod +x startNvr.sh

# ${port}：开启的端口号
# ${nums}：从${port}开始创建nums+1个视频流
# 可用的端口范围是${port}~${port}+${nums}
# 例如./startNvr.sh 1220 5 开启1220~1225个rtsp端口
./startNvr.sh ${port} ${nums}

#检查是否起流，若出现./live555MediaServer xxxx说明正常起流
ps -ef | grep live555MediaServer
```

**步骤3** 模型转换

下载模型，模型连接 https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/TSM/online_models.zip

将下载好的模型放在“/TSM/model”目录下。

将模型转换为onnx模型，在“/TSM/model/”目录下，运行脚本生成onnx模型

```shell
python3 pth2onnx1.py
```

将模型转换为om模型，在“/TSM/model/”目录下，运行脚本生成om模型

```shell
bash onnx2om1.sh
```

**步骤4** 程序测试

```shell
python3 online_infer.py
```

修改参数，'rtsp://192.168.88.110:1240/jester.264'为测试视频流

```python
def video2img():
    cmd = 'ffmpeg  -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 0 \"{}/img_%05d.jpg\"'.format('rtsp://192.168.88.110:1240/jester.264', './image')
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

## 5 软件依赖说明

如果涉及第三方软件依赖，请详细列出。

| 依赖软件   | 版本     | 说明    |
|--------|--------|-------|
| ffmpeg | 3.4.8  | 视频抽帧  |
| torch  | 1.10.0 | 数据集构建 |

官网下载地址：https://ffmpeg.org/download.html#build-linux

```Shell
xz -d ffmpeg-git-amd64-static.tar.xz
tar -xvf ffmpeg-git-amd64-static.tar
cd ./ffmpeg-git-20190424-amd64-static
./ffmpeg
```