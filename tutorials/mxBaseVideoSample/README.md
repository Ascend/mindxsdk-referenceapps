# 基于MxBase 的yolov3视频流推理样例

## 介绍

本开发样例是基于mxBase开发的端到端推理的C++应用程序，可在昇腾芯片上实现视频流的目标检测，并把可视化结果保存到本地。

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
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
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

main.cpp文件中，添加模型路径与 rtsp 流源地址（需要自行准备可用的视频流，视频流格式为H264）

[Live555拉流教程](../../docs/参考资料/Live555离线视频转RTSP说明文档.md)

```
...
initParam.modelPath = "{yolov3模型路径}";
...
 std::string streamName = "rtsp_Url";
```

VideoProcess.cpp文件中，设置视频的宽高值

```
const uint32_t VIDEO_WIDTH = {视频宽度};
const uint32_t VIDEO_HEIGHT = {视频高度};
```

### 配置环境变量

```
# 执行如下命令，打开.bashrc文件
vi .bashrc
# 在.bashrc文件中添加以下环境变量
MX_SDK_HOME=${SDK安装路径}
FFMPEG_PATH=${FFMPEG安装路径} 
# 若环境中没有安装ffmpeg，请联系支撑人员

LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/ascend-toolkit/:/usr/local/python3.9.2/lib:${FFMPEG_PATH}/lib
# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

#查看环境变量
env
```

### 配置CMakeLists

配置CMakeLists.txt文件中的`MX_SDK_HOME`与`FFMPEG_PATH`环境变量

```
set(MX_SDK_HOME {SDK实际安装路径})
set(FFMPEG_PATH {ffmpeg安装路径})
```

### 编译项目文件

新建立build目录，进入build执行cmake ..（..代表包含CMakeLists.txt的源文件父目录），在build目录下生成了编译需要的Makefile和中间文件。执行make构建工程，构建成功后就会生成可执行文件。

```
mkdir build

cd build

cmake ..

make -j
Scanning dependencies of target stream_pull_test
[ 25%] Building CXX object CMakeFiles/stream_pull_test.dir/main.cpp.o
[ 50%] Building CXX object CMakeFiles/stream_pull_test.dir/VideoProcess/VideoProcess.cpp.o
[ 75%] Building CXX object CMakeFiles/stream_pull_test.dir/Yolov3Detection/Yolov3Detection.cpp.o
[100%] Linking CXX executable ../stream_pull_test
[100%] Built target stream_pull_test

# stream_pull_test就是CMakeLists文件中指定生成的可执行文件。
```

### 执行脚本

执行run.sh脚本前请先确认可执行文件stream_pull_test已生成。

```
chmod +x run.sh
bash run.sh
```

### 查看结果

执行run.sh完毕后，可视化结果会被保存在工程目录下result文件夹中。

