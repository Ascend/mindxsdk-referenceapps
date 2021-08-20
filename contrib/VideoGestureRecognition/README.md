# 基于MxBase的视频手势识别运行

## 介绍

手势识别是指对视频中出现的手势进行分类,实现对本地（H264或H265）进行手势识别并分类，生成可视化结果。

### 目录结构
```
.
|-------- BlockingQueue
|           |---- BlockingQueue.h                   // 阻塞队列 (视频帧缓存容器)
|-------- ImageResizer
|           |---- ImageResizer.cpp                  // 图片缩放.cpp
|           |---- ImageResizer.h                    // 图片缩放.h
|-------- data
|           |---- gesture_test1.264                  // 手势识别测试数据1 (本地测试自行准备)
|           |---- gesture_test2.264                  // 手势识别测试数据2 (本地测试自行准备)
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
|-------- build.sh                                  // 样例编译脚本
|-------- CMakeLists.txt                            // CMake配置
|-------- main.cpp                                  // 视频手势识别测试样例
|-------- README.md                                 // ReadMe
|-------- run.sh                                    // 样例运行脚本

```

### 依赖
| 依赖软件      | 版本   | 下载地址                                                     | 说明                                         |
| ------------- | ------ | ------------------------------------------------------------ | -------------------------------------------- |
| ffmpeg        | 4.2.1  | [Link](https://github.com/FFmpeg/FFmpeg/archive/n4.2.1.tar.gz) | 视频转码解码组件                             |

**注意：**

第三方库默认全部安装到/usr/local/下面，全部安装完成后，请设置环境变量
```bash
export PATH=/usr/local/ffmpeg/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH
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

> 模型转换

步骤1 在

**步骤1** 在ModelZoo上下载Resnet18模型。[下载地址](https://gitee.com/ascend/samples/tree/master/python/contrib/gesture_recognition_picture#https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/gesture_recognition/ATC_gesture_recognition_Caffe_AE)

**步骤2** 将获取到的Resnet18模型pb文件存放至："样例项目所在目录/model/"，同时下载cfg文件：[下载方式]（wget https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesture_recognition/insert_op.cfg）

**步骤3** 模型转换

在pb文件所在目录下执行一下命令

```
# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 执行，转换Resnet18模型
# Execute, transform Resnet18 model.

atc --model=./resnet18_gesture.prototxt --weight=./resnet18_gesture.caffemodel --framework=0 --output=gesture_yuv --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --input_shape="data:1,3,224,224" --input_format=NCHW
```

执行完模型转换脚本后，会生成相应的.om模型文件。 执行完模型转换脚本后，会生成相应的.om模型文件。

模型转换使用了ATC工具，如需更多信息请参考:

 https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html

> 相关参数修改

main.cpp中配置rtsp流源地址(需要自行准备可用的视频流，视频流格式为H264或H265)

[Live555拉流教程](../../docs/参考资料/Live555离线视频转RTSP说明文档.md)
```c++ rtspList.emplace_back("#{本地或rtsp流地址}"); ```

[配置ResnetDetector插件的模型加载路径modelPath]
```c++ reasonerConfig.resnetModelPath = "${Resnet18.om模型路径}"```

配置ResnetDetector插件的模型加载路径labelPath
```c++ reasonerConfig.resnetLabelPath = "${resnet18.names路径}";```

其他可配置项DECODE_FRAME_QUEUE_LENGTH DECODE_FRAME_WAIT_TIME SAMPLING_INTERVAL MAX_SAMPLING_INTERVAL
```c++ DECODE_FRAME_QUEUE_LENGTH = 100; DECODE_FRAME_WAIT_TIME = 10; SAMPLING_INTERVAL = 24; MAX_SAMPLING_INTERVAL = 100;```


### 配置环境变量

```
# 执行如下命令，打开.bashrc文件
vi .bashrc
# 在.bashrc文件中添加以下环境变量
MX_SDK_HOME=${SDK安装路径}
FFMPEG_HOME=${FFMPEG安装路径}

LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${FFMPEG_HOME}/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/

GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner

GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

#查看环境变量
env
```

### 配置SDK路径

配置CMakeLists.txt文件中的`MX_SDK_HOME`与`FFMPEG_PATH`环境变量

```
set(MX_SDK_HOME ${SDK安装路径}/mxVision)
set(FFMPEG_PATH {ffmpeg实际安装路径})
```


### 编译项目文件

手动编译请参照 ①，脚本编译请参照 ②

>  ① 新建立build目录，进入build执行cmake ..（..代表包含CMakeLists.txt的源文件父目录），在build目录下生成了编译需要的Makefile和中间文件。执行make构建工程，构建成功后就会生成可执行文件。

```
mkdir build

cd build

cmake ..

-- The C compiler identification is GNU 7.5.0
-- The CXX compiler identification is GNU 7.5.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done


make

Scanning dependencies of target sample
[ 11%] Building CXX object CMakeFiles/sample.dir/main.cpp.o
[ 22%] Building CXX object CMakeFiles/sample.dir/StreamPuller/StreamPuller.cpp.o
[ 33%] Building CXX object CMakeFiles/sample.dir/VideoDecoder/VideoDecoder.cpp.o
[ 44%] Building CXX object CMakeFiles/sample.dir/ImageResizer/ImageResizer.cpp.o
[ 66%] Building CXX object CMakeFiles/sample.dir/ResnetDetector/ResnetDetector.cpp.o
[ 77%] Building CXX object CMakeFiles/sample.dir/MultiChannelVideoReasoner/MultiChannelVideoReasoner.cpp.o
[ 88%] Building CXX object CMakeFiles/sample.dir/FrameSkippingSampling/FrameSkippingSampling.cpp.o
[100%] Linking CXX executable ../sample
[100%] Built target sample
# sample就是CMakeLists文件中指定生成的可执行文件。
```

>  ② 运行项目根目录下的`build.sh`
```bash
chmod +x build.sh
bash build.sh
```
### 执行脚本

执行`run.sh`脚本前请先确认可执行文件`sample`已生成。

```
chmod +x run.sh
bash run.sh
```

### 查看结果

执行`run.sh`完毕后，如果配置了检测结果写文件，sample会将目标检测结果保存在工程目录下`result`中。