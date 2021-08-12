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
|                   |---- PerformanceMonitor.cpp    // 性能监控.cpp
|                   |---- PerformanceMonitor.h      // 性能监控.h
|           |---- Util.h                            // 工具类
|-------- VideoDecoder
|           |---- VideoDecoder.cpp                  // 视频解码.cpp
|           |---- VideoDecoder.h                    // 视频解码.h
|-------- YoloDetector
|           |---- YoloDetector.cpp                  // Yolo检测.cpp
|           |---- YoloDetector.h                    // Yolo检测.h
|-------- CMakeLists.txt                            // CMake配置
|-------- main.cpp                                  // 多路视频推理测试样例
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

下载完解压，按以下命令编译即可

```bash
./configure --prefix=/usr/local/ffmpeg --enable-shared
make -j
make install
```

### 准备工作

> 模型转换

**步骤1** 在ModelZoo上下载YOLOv3模型 ，选择“历史版本”中版本1.1下载。[下载地址](https://www.hiascend.com/zh/software/modelzoo/detail/C/210261e64adc42d2b3d84c447844e4c7)

**步骤2** 将获取到的YOLOv3模型pb文件存放至："样例项目所在目录/model/"。

**步骤3** 模型转换

在pb文件所在目录下执行以下命令

```
# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 执行，转换YOLOv3模型
# Execute, transform YOLOv3 model.

atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input/input_data:1,416,416,3" --out_nodes="conv_lbbox/BiasAdd:0;conv_mbbox/BiasAdd:0;conv_sbbox/BiasAdd:0"
# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
```

执行完模型转换脚本后，会生成相应的.om模型文件。 执行完模型转换脚本后，会生成相应的.om模型文件。

模型转换使用了ATC工具，如需更多信息请参考:

 https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html

> 相关参数修改

main.cpp中配置rtsp流源地址(需要自行准备可用的视频流，视频流格式为H264或H265)

[Live555拉流教程](../../docs/参考资料/Live555离线视频转RTSP说明文档.md)
```c++
rtspList.emplace_back("#{rtsp流地址1}");
rtspList.emplace_back("#{rtsp流地址2}");
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
```c++
reasonerConfig.maxDecodeFrameQueueLength = 400; // 多路视频时请适当减小
reasonerConfig.writeDetectResultToFile = true; // 检测结果是否写文件
reasonerConfig.enablePerformanceMonitorPrint = true; // 性能可视化开关
reasonerConfig.intervalPerformanceMonitorPrint = 5; // 性能监控输出间隔(s)
```

### 配置环境变量

```bash
# 执行如下命令，打开.bashrc文件
cd $HOME
vi .bashrc
# 在.bashrc文件中添加以下环境变量
MX_SDK_HOME=${SDK安装路径}
FFMPEG_PATH=${FFMPEG安装路径}

LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${FFMPEG_PATH}/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/

GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner

GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

#查看环境变量
env
```

### 配置CMakeLists

配置CMakeLists.txt文件中的`MX_SDK_HOME`与`FFMPEG_PATH`环境变量
> Tips: 如果安装FFmpeg的过程中选择了`--enable-libx264`选项，还需要添加`X264_HOME`环境变量
- eg: `MX_SDK_HOME ../../MindXSDK/mxVision` `FFMPEG_PATH ../../local/ffmpeg` `X264_HOME ../../local/x264`

```cmake
set(MX_SDK_HOME {MindXSDK安装路径})
set(FFMPEG_PATH {ffmpeg实际安装路径})
set(X264_HOME {x264实际安装路径}) # 可选
```

### 编译项目文件

新建立build目录，进入build执行cmake ..（..代表包含CMakeLists.txt的源文件父目录），在build目录下生成了编译需要的Makefile和中间文件。执行make构建工程，构建成功后就会生成可执行文件。

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
Scanning dependencies of target sample
[ 12%] Building CXX object CMakeFiles/sample.dir/main.cpp.o
[ 25%] Building CXX object CMakeFiles/sample.dir/StreamPuller/StreamPuller.cpp.o
[ 37%] Building CXX object CMakeFiles/sample.dir/VideoDecoder/VideoDecoder.cpp.o
[ 50%] Building CXX object CMakeFiles/sample.dir/ImageResizer/ImageResizer.cpp.o
[ 62%] Building CXX object CMakeFiles/sample.dir/YoloDetector/YoloDetector.cpp.o
[ 75%] Building CXX object CMakeFiles/sample.dir/MultiChannelVideoReasoner/MultiChannelVideoReasoner.cpp.o
[ 87%] Building CXX object CMakeFiles/sample.dir/Util/PerformanceMonitor/PerformanceMonitor.cpp.o
[100%] Linking CXX executable ../sample
[100%] Built target sample
# sample就是CMakeLists文件中指定生成的可执行文件。
```

### 执行脚本

执行run.sh脚本前请先确认可执行文件sample已生成。

```
chmod +x run.sh
bash run.sh
```

### 查看结果

执行run.sh完毕后，如果配置了检测结果写文件，sample会将目标检测结果保存在工程目录下`result`中。