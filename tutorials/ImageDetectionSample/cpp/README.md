# 图像检测样例命令行运行

## 介绍

提供的sample样例，实现对本地图片进行YOLOv3目标检测，生成可视化结果。

### 准备工作

> 模型转换

步骤1 在

**步骤1** 在ModelZoo上下载YOLOv3模型。[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ActionRecognition/ATC%20YOLOv3%28FP16%29%20from%20TensorFlow%20-%20Ascend310.zip)

**步骤2** 将获取到的YOLOv3模型pb文件和coco.names存放至拷贝到model目录下。

**步骤3** 模型转换
设置环境变量（请确认ascend_toolkit_path路径是否正确）

```
. ${ascend_toolkit_path}/set_env.sh
```

执行atc命令，转换YOLOv3模型
```
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"

# 说明1：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
# 说明2：若用例执行在310B上，则--soc_version=Ascend310需修改为Ascend310B1
```

执行完模型转换脚本后，会生成相应的.om模型文件。 执行完模型转换脚本后，会生成相应的.om模型文件。

模型转换使用了ATC工具，如需更多信息请参考:

 https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md

> 配置pipeline

配置mxpi_tensorinfer插件的模型加载路径`modelPath`

```
"mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "${yolov3.om模型路径}"
            },
            "factory": "mxpi_tensorinfer",
            "next": "mxpi_objectpostprocessor0"
        },
```

配置模型后处理插件mxpi_objectpostprocessor，`postProcessLibPath`的后处理库路径，路径根据SDK安装路径决定，可以通过`find -name libyolov3postprocess.so`搜索路径。

- eg: SDK安装路径/mxVision/lib/modelpostprocessors/libyolov3pos

```
"mxpi_objectpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "model/yolov3_tf_bs1_fp16.cfg",
                "labelPath": "${SDK安装路径}/samples/mxVision/models/yolov3/yolov3.names",
                "postProcessLibPath": "${SDK安装路径}/lib/modelpostprocessors/libyolov3postprocess.so"
            },
            "factory": "mxpi_objectpostprocessor",
            "next": "appsink0"
        },
```

### 配置环境变量

```
# 执行如下命令，打开.bashrc文件
vi .bashrc
# 在.bashrc文件中添加以下环境变量
MX_SDK_HOME=${SDK安装路径}

LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/

GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner

GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

#查看环境变量
env
```

### 配置SDK路径

配置CMakeLists.txt文件中的`MX_SDK_HOME`环境变量

```
set(MX_SDK_HOME ${SDK安装路径}/mxVision)
```

### 编译项目文件

新建立build目录，进入build执行cmake ..（..代表包含CMakeLists.txt的源文件父目录），在build目录下生成了编译需要的Makefile和中间文件。执行make构建工程，构建成功后就会生成可执行文件。

```
mkdir build

cd build

cmake ..

make
Scanning dependencies of target sample
[ 50%] Building CXX object CMakeFiles/sample.dir/main.cpp.o
[100%] Linking CXX executable ../sample
[100%] Built target sample
# sample就是CMakeLists文件中指定生成的可执行文件。
```

### 执行脚本
准备一张待检测图片，放到项目目录下命名为test.jpg
执行run.sh脚本前请先确认可执行文件sample已生成，并给脚本添加可执行权限。

```
bash run.sh
```

### 查看结果

执行run.sh完毕后，sample会将目标检测结果保存在工程目录下`result.jpg`中。