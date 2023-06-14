# 基于MxBase的多路视频检测

## 介绍

多路视频检测，实现同时对两路本地视频或RTSP视频流(H264或H265)进行YOLOv3目标检测，生成可视化结果(可选)。

### 目录结构
```
.
|-------- BlockingQueue
|           |---- BlockingQueue.h                   // 阻塞队列 (视频帧缓存容器)
|-------- data
|           |---- test_video_1.264                  // 多路视频测试数据1 (本地测试自行准备)
|           |---- test_video_2.264                  // 多路视频测试数据2 (本地测试自行准备)
|-------- ImageResizer
|           |---- ImageResizer.cpp                  // 图片缩放.cpp
|           |---- ImageResizer.h                    // 图片缩放.h
|-------- model
|           |---- aipp_yolov3_416_416.aippconfig    // yolov3 模型转换配置文件
|           |---- coco.names                        // yolov3 标签文件
|-------- MultiChannelVideoReasoner
|           |---- MultiChannelVideoReasoner.cpp     // 多路视频推理业务逻辑封装.cpp
|           |---- MultiChannelVideoReasoner.h       // 多路视频推理业务逻辑封装.h
|-------- result                                    // 视频推理结果存放处
|-------- StreamPuller
|           |---- StreamPuller.cpp                  // 视频拉流.cpp
|           |---- StreamPuller.h                    // 视频拉流.h
|-------- Util
|           |---- PerformanceMonitor
|                   |---- PerformanceMonitor.cpp    // 性能管理.cpp
|                   |---- PerformanceMonitor.h      // 性能管理.h
|           |---- Util.cpp                          // 工具类.cpp
|           |---- Util.h                            // 工具类.h
|-------- VideoDecoder
|           |---- VideoDecoder.cpp                  // 视频解码.cpp
|           |---- VideoDecoder.h                    // 视频解码.h
|-------- YoloDetector
|           |---- YoloDetector.cpp                  // Yolo检测.cpp
|           |---- YoloDetector.h                    // Yolo检测.h
|-------- build.sh                                  // 样例编译脚本
|-------- CMakeLists.txt                            // CMake配置
|-------- main.cpp                                  // 多路视频推理测试样例
|-------- README.md                                 // ReadMe
|-------- run.sh                                    // 样例运行脚本

```
### 依赖
ffmpeg 4.2.1
mxVision 5.0.RC1
Ascend-CANN-toolkit (310使用6.3.RC1, 310B使用6.2.RC1)

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

**步骤1** 下载YOLOv3模型 。[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ActionRecognition/ATC%20YOLOv3%28FP16%29%20from%20TensorFlow%20-%20Ascend310.zip)

**步骤2** 将获取到的YOLOv3模型pb文件存放至："样例项目所在目录/model/"。

**步骤3** 模型转换

在pb文件所在目录下执行以下命令

```
# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 执行，转换YOLOv3模型
# Execute, transform YOLOv3 model.

atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"
# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
# 注意：若推理芯片为310B，需将atc-env脚本中模型转换atc命令中的soc_version参数设置为Ascend310B1。
```

执行完模型转换脚本后，会生成相应的.om模型文件。
模型转换使用了ATC工具，如需更多信息请参考:

 https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md

> 相关参数修改

main.cpp中配置rtsp流源地址(需要自行准备可用的视频流，视频流格式为H264或H265)

[Live555拉流教程](../../docs/参考资料/Live555离线视频转RTSP说明文档.md)
```c++
rtspList.emplace_back("#{rtsp流地址1}");
rtspList.emplace_back("#{rtsp流地址2}");
```

配置config sdk路径
```c++
configUtil.LoadConfiguration("${MindXSDK安装路径}/config/logging.conf", configData, MxBase::ConfigMode::CONFIGFILE);
```

配置YoloDetector插件的模型加载路径`modelPath`
```c++
reasonerConfig.yoloModelPath = "${yolov3.om模型路径}"
```

配置YoloDetector插件的模型加载路径`labelPath`

- eg: SDK安装路径/samples/mxVision/models/yolov3/coco.names
```c++
reasonerConfig.yoloLabelPath = "${yolov3 coco.names路径}";
```

其他可配置项`maxDecodeFrameQueueLength` `writeDetectResultToFile` `enablePerformanceMonitorPrint` `intervalPerformanceMonitorPrint` 
`intervalMainThreadControlCheck` `printDetectResult` `enableIndependentThreadForEachDetectStep`
> 开启 `enableIndependentThreadForEachDetectStep` 时请减小 `maxDecodeFrameQueueLength` 的值，建议最大值 `200`  
> 同时开启 `writeDetectResultToFile`时请进一步减小 `maxDecodeFrameQueueLength` 的值，建议最大值 `100`
```c++
reasonerConfig.maxDecodeFrameQueueLength = 400; // 多路视频时请适当减小
reasonerConfig.writeDetectResultToFile = true; // 检测结果是否写文件, 默认为false
reasonerConfig.enablePerformanceMonitorPrint = false; // 性能可视化开关，默认为false
reasonerConfig.intervalPerformanceMonitorPrint = 5; // 性能管理输出间隔(s)
reasonerConfig.intervalMainThreadControlCheck = 2; // 流程检查间隔(ms)
reasonerConfig.printDetectResult = true; // 输出检测结果，默认为true
reasonerConfig.enableIndependentThreadForEachDetectStep = true; // 为每个检测步骤启用独立线程，默认为true
```

### 配置环境变量

```shell
. /usr/local/Ascend/ascend-toolkit/set_envv.sh # Ascend-cann-toolkit开发套件包默认安全路径，根据实际安装路径修改
. ${MX_SDK_HOME}/mxVision/set_env.sh # ${MX_SDK_HOME}替换为用户的SDK安装路径shell
```

### 配置CMakeLists

配置CMakeLists.txt文件中的`MX_SDK_HOME`与`FFMPEG_PATH`环境变量
> Tips: 如果安装FFmpeg的过程中选择了`--enable-libx264`选项，还需要添加`X264_HOME`环境变量
- eg: `MX_SDK_HOME ../../MindXSDK/mxVision` `FFMPEG_PATH ../../local/ffmpeg` `X264_HOME ../../local/x264`

```cmake
set(MX_SDK_HOME {MindXSDK安装路径})
set(FFMPEG_HOME {ffmpeg实际安装路径})
set(X264_HOME {x264实际安装路径}) # 可选
```

### 编译项目文件

手动编译请参照 ①，脚本编译请参照 ②

> ① 新建build目录，进入build执行cmake ..（..代表包含CMakeLists.txt的源文件父目录），在build目录下生成了编译需要的Makefile和中间文件。执行make构建工程，构建成功后就会生成可执行文件。

```bash
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
Scanning dependencies of target multiChannelVideoReasoner
[ 44%] Building CXX object CMakeFiles/multiChannelVideoReasoner.dir/main.cpp.o
[ 44%] Building CXX object CMakeFiles/multiChannelVideoReasoner.dir/VideoDecoder/VideoDecoder.cpp.o
[ 44%] Building CXX object CMakeFiles/multiChannelVideoReasoner.dir/ImageResizer/ImageResizer.cpp.o
[ 44%] Building CXX object CMakeFiles/multiChannelVideoReasoner.dir/YoloDetector/YoloDetector.cpp.o
[ 55%] Building CXX object CMakeFiles/multiChannelVideoReasoner.dir/StreamPuller/StreamPuller.cpp.o
[ 66%] Building CXX object CMakeFiles/multiChannelVideoReasoner.dir/Util/PerformanceMonitor/PerformanceMonitor.cpp.o
[ 77%] Building CXX object CMakeFiles/multiChannelVideoReasoner.dir/Util/Util.cpp.o
[ 88%] Building CXX object CMakeFiles/multiChannelVideoReasoner.dir/MultiChannelVideoReasoner/MultiChannelVideoReasoner.cpp.o
[100%] Linking CXX executable ../multiChannelVideoReasoner
[100%] Built target multiChannelVideoReasoner
# sample就是CMakeLists文件中指定生成的可执行文件。
```

> ② 运行项目根目录下的`build.sh`
```bash
chmod +x build.sh
bash build.sh
```
### 执行脚本

执行`run.sh`脚本前请先确认可执行文件`multiChannelVideoReasoner`已生成。

```
chmod +x run.sh
bash run.sh
```

### 查看结果

执行`run.sh`完毕后，如果配置了检测结果写文件，sample会将目标检测结果保存在工程目录下`result`中。

## 疑难解惑
本工程中需要使用cann下dvpp的相关组件，CMake中配置为指向默认安装的/usr/local/Ascend/ascend-toolkit/latest/include，如用户有变更应该更改为实际位置。