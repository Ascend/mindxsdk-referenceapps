# 基于MxBase的视频手势识别运行

## 介绍

手势识别是指对视频中出现的手势进行分类，实现对本地（H264）进行手势识别并分类，生成可视化结果。
使用测试视频中的手势尺寸大致应为视频大小的二分之一，同时应当符合国际标准，背景要单一，手势要清晰，光线充足；视频切勿有遮挡，不清晰等情况。

## 支持的产品

昇腾Atlas 500 A2

### 目录结构
```
.
|-------- BlockingQueue
|           |---- BlockingQueue.h                   // 阻塞队列 (视频帧缓存容器)
|-------- ImageResizer
|           |---- ImageResizer.cpp                  // 图片缩放.cpp
|           |---- ImageResizer.h                    // 图片缩放.h
|-------- FrameSkippingSampling
|           |---- FrameSkippingSampling.cpp         // 跳帧采样.cpp
|           |---- FrameSkippingSampling.h           // 跳帧采样.h
|-------- model
|           |---- resnet18.cfg                      // Resnet18 模型转换配置文件
|           |---- resnet18.names                    // Resnet18 标签文件
|-------- VideoGestureReasoner
|           |---- VideoGestureReasoner.cpp          // 视频手势推理业务逻辑封装.cpp
|           |---- VideoGestureReasoner.h            // 视频手势推理业务逻辑封装.h
|-------- result                                    // 推理结果存放处（图片的形式）
|-------- StreamPuller
|           |---- StreamPuller.cpp                  // 视频拉流.cpp
|           |---- StreamPuller.h                    // 视频拉流.h
|-------- Util
|           |---- Util.cpp                          // 工具类.cpp
|           |---- Util.h                            // 工具类.h
|-------- VideoDecoder
|           |---- VideoDecoder.cpp                  // 视频解码.cpp
|           |---- VideoDecoder.h                    // 视频解码.h
|-------- ResnetDetector
|           |---- ResnetDetector.cpp                  // Resnet识别.cpp
|           |---- ResnetDetector.h                    // Resnet识别.h
|-------- data
|-------- build.sh                                  // 样例编译脚本
|-------- CMakeLists.txt                            // CMake配置
|-------- main.cpp                                  // 视频手势识别测试样例
|-------- README.md                                 // ReadMe
|-------- run.sh                                    // 样例运行脚本

```

### 依赖

推荐系统为ubuntu 18.04。

| 软件名称 | 版本   |
| -------- | ------ |
| python    | 3.9.2     | 
| MindX SDK     |    5.0RC1    |
| CANN | 310使用6.3.RC1<br>310B使用6.2.RC1 |


| 第三方依赖软件      | 版本   | 下载地址                                                     | 说明                                         |
| ------------- | ------ | ------------------------------------------------------------ | -------------------------------------------- |
| ffmpeg        | 4.2.1  | [Link](https://github.com/FFmpeg/FFmpeg/archive/n4.2.1.tar.gz) | 视频转码解码组件                             |

**注意：**

第三方库默认全部安装到/usr/local/下面，全部安装完成后，请设置环境变量
```bash
export PATH=/usr/local/ffmpeg/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH
export FFMPEG_PATH=/usr/local/ffmpeg/
```

#### FFmpeg

下载完，按以下命令进行解压和编译

```bash
tar -xzvf n4.2.1.tar.gz
cd FFmpeg-n4.2.1
./configure --prefix=/usr/local/ffmpeg --enable-shared
make -j
make install
```

### 准备工作
> 配置环境变量

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh #toolkit默认安装路径，根据实际安装路径修改
. ${SDK_INSTALL_PATH}/mxVision/set_env.sh
```

> 模型转换

**步骤1** 下载Resnet18模型权重和网络以及cfg文件。[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/VideoGestureRecognition/model.zip)

**步骤2** 将获取到的文件存放至："样例项目所在目录/model/"

**步骤3** 模型转换

在模型权重和网络文件所在目录下执行以下命令

```
# 执行，转换Resnet18模型
# Execute, transform Resnet18 model.

atc --model=./resnet18_gesture.prototxt --weight=./resnet18_gesture.caffemodel --framework=0 --output=gesture_yuv --soc_version=Ascend310B1 --insert_op_conf=./insert_op.cfg --input_shape="data:1,3,224,224" --input_format=NCHW
```

执行完模型转换脚本后，会生成相应的.om模型文件。 执行完模型转换脚本后，会生成相应的.om模型文件。

模型转换使用了ATC工具，如需更多信息请参考:

 https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md

> 相关参数修改

main.cpp中配置rtsp流源地址(需要自行准备可用的视频流，视频流格式为H264)。
同样地测试视频也可下载（[链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/VideoGestureRecognition/data.zip)）。

```rtspList.emplace_back("#{本地或rtsp流地址}"); ```

提示：使用测试视频中的手势尺寸大致应为视频大小的二分之一，同时应当符合国际标准，背景要单一，手势要清晰，光线充足；视频切勿有遮挡，不清晰等情况。

[Live555拉流教程](../../docs/参考资料/Live555离线视频转RTSP说明文档.md)

配置ResnetDetector插件的模型加载路径modelPath
```reasonerConfig.resnetModelPath = "${Resnet18.om模型路径}"```

配置ResnetDetector插件的模型加载路径labelPath
```reasonerConfig.resnetLabelPath = "${resnet18.names路径}";```

其他可配置项DECODE_FRAME_QUEUE_LENGTH DECODE_FRAME_WAIT_TIME SAMPLING_INTERVAL MAX_SAMPLING_INTERVAL
```
DECODE_FRAME_QUEUE_LENGTH = 100; 
DECODE_FRAME_WAIT_TIME = 10; 
SAMPLING_INTERVAL = 24; 
MAX_SAMPLING_INTERVAL = 100;
```

### 配置SDK路径

配置CMakeLists.txt文件中的`FFMPEG_PATH`环境变量

```
set(FFMPEG_PATH {ffmpeg实际安装路径})

```


### 编译项目文件

手动编译请参照 ①，脚本编译请参照 ②

>  ① 新建立build目录，进入build执行cmake ..（..代表包含CMakeLists.txt的源文件父目录），在build目录下生成了编译需要的Makefile和中间文件。执行make构建工程，构建成功后就会生成可执行文件。

```
mkdir build

cd build

cmake ..

make

....

[100%] Linking CXX executable ../sample
[100%] Built target sample
# sample就是CMakeLists文件中指定生成的可执行文件。
```

>  ② 运行项目根目录下的`build.sh`
```bash
bash build.sh
```
### 执行脚本

执行`run.sh`脚本前请先确认可执行文件`sample`已生成。

```
bash run.sh
```

### 查看结果

执行`run.sh`完毕后，如果配置了检测结果写文件，sample会将手势识别结果，以jpg格式的图片保存在工程目录下`result`中。