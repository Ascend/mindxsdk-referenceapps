# 路面分割
## 1. 简介
本样例基于MindX SDK实现了端到端的路面分割功能，主要采用了Unet模型对输入的路面图片进行语义分割，输出mask掩膜，然后与原图结合，生成路面语义分割后的可视化结果。
## 2. 目录结构
```
├── config #配置文件目录
│   └── aipp_road_segmentation.config
├── model  #模型目录
│	└──  Road.onnx
│   └──  road_segmentation.om
│  	└──  pt2onnx.py  
├── pipeline
│   └── road.pipeline
├── plugin #后处理插件目录
│ 	└──RoadSegPostProcess
│		├── build
│		├── build.sh  #编译脚本
│		├── lib 
│		│     └──plugins #编译好的插件存放位置
│ 		├── CMakeLists.txt
│   	├── MxpiRoadSegPostProcess.cpp
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
|pytorch|>= 1.5.0|
|CANN|5.0.4|
|C++| 11.0|
|opencv2| |> 

## 4. 模型转换  
### 4.1 导出onnx文件
  获取[路面分割案例](https://github.com/tunafatih/Road-Free-Space-Segmentation-Internship-Project)，在本地使用pt2onnx.py文件，将pt权重文件转换成onnx文件，或可[点击此处](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/RoadSegmentation/model.zip)下载转换好的onnx文件。
### 4.2 使用Ascend atc工具将onnx模型转换为om模型
在使用[atc工具](https://www.hiascend.com/document/detail/zh/canncommercial/504/inferapplicationdev/atctool)之前需配置环境
```
# CANN安装目录
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
```
之后将4.1节中导出的onnx文件上传至```./model```目录下，在该目录下执行
```
atc --framework=5 --model=Road.onnx --output=road_segmentation --input_format=NCHW  --insert_op_conf=aipp_road_segmentation.config --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310  
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
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:/home/weifeng1/MindX_SDK/mxVision/lib/plugins
export PYTHONPATH=${MX_SDK_HOME}/python:${PYTHONPATH}

#查看环境变量
env
```
### 5.2 编译后处理插件
在样例目录下执行
```
./build.sh
```
或在RoadSegPostProcess目录下执行
```
mkdir build
cd build
cmake ..
make -j
#之后可以在 样例/plugin/lib/plugins目录下看到编译好的.so插件
#进入/plugin/lib/plugins目录
chmod 640 *.so #修改权限
然后编译好的插件移动到${SDK安装路径}/lib/plugins目录下
```
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
可修改main.py文件中的FILE_PATH的变量值将其改为测试图片的地址
### 5.5 输出结果
在样例目录下执行
```
python3.9 main.py
```
或
```
./run.sh
```
可看到路面分割可视化结果 
## 6 结果展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/3a07ceda71b9402b88cfd38a9e033622.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/701e66974ec8460091ba7ee96426423c.png)



## 7 约束限制
输入图片的宽高须是偶数且图片只支持Huffman编码，[imagedecoder插件介绍](https://www.hiascend.com/document/detail/zh/mind-sdk/204/vision/mxvisionug/mxvisionug_0115.html)和[imageencoder插件介绍](https://www.hiascend.com/document/detail/zh/mind-sdk/204/vision/mxvisionug/mxvisionug_0120.html)
![在这里插入图片描述](https://img-blog.csdnimg.cn/a90335b846484d76992d07473c1dea3d.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/93b777389c4743b9a2783ae6e4070b31.png)








