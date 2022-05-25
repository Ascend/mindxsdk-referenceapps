# 路面分割
## 1. 简介
本样例基于MindX SDK实现了端到端的路面分割功能，主要采用了Unet模型对输入的路面图片进行语义分割，输出mask掩膜，然后与原图结合，生成标注出路面的可视化结果。
## 2. 目录结构
```
├── config #配置文件目录
│   └── aipp_road_segmentation.config
├── model  #模型目录
│ 	└──Road.onnx 
├── pipeline
│   └── road.pipeline
├── plugin #后处理插件目录
│ 	└──RoadSegPostProcess
│		├── build
│		├── build.sh  #编译脚本
│		├── lib 
│		│     └──plugins #编译好的插件存放位置
│ 		├── CMakeLists.txt
│   		├── MxpiRoadSegPostProcess.cpp
│  		└── MxpiRoadSegPostProcess.cpp.h
├── main.py
├── README.md
├── build.sh
└── run.sh
```
## 3. 依赖
| 软件名称 | 版本   |
| :--------: | :------: |
|ubantu 18.04|18.04.1 LTS   |
|MindX SDK|2.0.4|
|Python|3.9.2|
|CANN|5.0.4|
|C++| 11.0|
|opencv2| |> 

## 4. 模型转换  
使用Ascend atc工具将onnx模型转换为om模型
```
# CANN安装目录
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 在 样例/model目录下执行atc转换命令 
atc --framework=5 --model=Road.onnx --output=road_segmentation --input_format=NCHW  --insert_op_conf=../config/aipp_road_segmentation.config --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310  
```
若出现以下信息，则转换成功
```
ATC run success
```
## 5. 推理
### 5.1 配置MindX SDK环境变量
```
export MX_SDK_HOME=${SDK安装路径}
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/MindX_SDK/mxVision/lib/plugins
export PYTHONPATH=${MX_SDK_HOME}/python:${PYTHONPATH}

#查看环境变量
env
```
### 5.2 编译后处理插件
```
在plugin目录下
mkdir build
cd build
cmake ..
make -j
```
之后可以在 样例/plugin/lib/plugins目录下看到编译好的.so插件
```
#进入该目录
chmod 640 *.so #修改权限
```
然后其移动到${SDK安装路径}/lib/plugins目录下

### 5.3 配置pipeline
根据所需场景，配置pipeline文件，调整路径参数等。
```
  #配置mxpi_tensorinfer插件的模型加载路径： modelPath
  "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "${road_segmentation.om模型路径}"
            },
            "factory": "mxpi_tensorinfer",
            "next": "appsink0"
        },
	#配置可视化结果输出路径：location
	"appsink0": {
            "props": {
                "blocksize": "4096000",
	"location":"${输出结果文件名}" 
            },
            "factory": "filesink"
        }
```
### 5.4 更改图片
可修改main.py文件中的filepath的变量值将其改为测试图片的地址
### 5.5 输出结果
```
#在样例目录下执行
python3.9 main.py
```
可看到路面分割可视化结果 




