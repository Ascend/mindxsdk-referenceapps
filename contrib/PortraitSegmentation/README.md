# MindXSDK 人像分割与背景替换

## 1 简介
  本开发样例基于MindX SDK实现了端到端的人像分割与背景替换（Portrait Segmentation and Background Replacement, PSBR）。PSBR的主要功能是使用Portrait模型对输入图片中的人像进行分割，然后与背景图像融合，实现背景替换。  
样例输入：带有简单背景的单人jpg图片和一张没有人像的背景jpg图片。  
样例输出：人像背景替换后的jpg图片。<br/>
## 2 目录结构
本工程名称为PortraitSegmentation，工程目录如下图所示：
```
|-------- models
|           |---- portrait.pb                 // 人像分割pb模型
|           |---- insert_op.cfg               // 模型转换配置文件
|           |---- portrait.om                 // 人像分割om模型
|-------- pipline
|           |---- segment.pipeline            // 人像分割模型流水线配置文件
|-------- main.py                             
|-------- README.md   
```
## 3 依赖

| 软件名称 | 版本   |
| :--------: | :------: |
|ubantu 18.04|18.04.1 LTS   |
|MindX SDK|2.0.4|
|Python| 3.9.2|
|numpy | 1.18.2 |
|opencv_python|3.4.0|

请注意MindX SDK使用python版本为3.9.2，如出现无法找到python对应lib库请在root下安装python3.9开发库  
`apt-get install libpython3.9`
## 4 模型转换
人体语义分割采用提供的Portrait.pb模型。由于原模型是基于tensorflow的人像分割模型，因此我们需要借助于ATC工具将其转化为对应的om模型。  
**步骤1**  下载Portrait原始模型与&ensp; 对应的cfg文件：[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PortraitSegmentation/model.zip)      

**步骤2**  将获取到的Portrait模型pb文件和cfg文件存放至：“项目所在目录/models”  

**步骤3**  模型转换  

在pb文件所在目录下执行以下命令  
```
#设置环境变量（请确认install_path路径是否正确）  
#Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest    

export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH 
 
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg  

export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH  
export ASCEND_OPP_PATH=${install_path}/opp    

#执行，转换Portrait.pb模型
#Execute, transform Portrait.pb model.
  
atc --model=portrait.pb  --input_shape="Inputs/x_input:1,224,224,3"  --framework=3  --output=portrait --insert_op_conf=insert_op.cfg --soc_version=Ascend310 
```
执行完模型转换脚本后，若提示如下信息说明模型转换成功，会在output参数指定的路径下生成portrait.om模型文件。  
```
ATC run success  
```
模型转换使用了ATC工具，如需更多信息请参考：  

https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/sdk/tutorials/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md

## 5 测试

1. 获取om模型   
```
见4： 模型转换
```
2. 配置
```   
#执行如下命令，打开.bashrc文件
cd $home
vi .bashrc
#在.bashrc文件中添加以下环境变量:

export MX_SDK_HOME=${SDK安装路径}/mxVision

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/

export PYTHONPATH=${MX_SDK_HOME}/python

export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner

export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

#保存退出.bashrc
#执行如下命令使环境变量生效
source ~/.bashrc

#查看环境变量
env
```
3. 配置pipeline  
根据所需场景，配置pipeline文件，调整路径参数等。
```
  #配置mxpi_tensorinfer插件的模型加载路径： modelPath
  "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "${portrait.om模型路径}"
            },
            "factory": "mxpi_tensorinfer",
            "next": "appsink0"
        },
```
4. 获取测试需要的测试图片  

在工程目录下新建data文件夹，进入data文件夹下载对应jpg格式的人像和背景测试图片，并分别命名为background.jpg以及portrait.jpg。


5. 在工程主目录下执行如下命令，新建result文件夹，用于存放推理结果

```
mkdir result
```

6. 运行可执行文件

切换至工程主目录，执行以下命令运行样例。命令行格式为 [python3 main.py 背景图片路径 人像图片路径 阈值参数]  
其中阈值参数的范围是闭区间[0,1]，从0到1，随着阈值参数增加，人像的广度也越高，即会将所有的人像部分进行背景替换。  
默认的阈值参数为1。

```
例：python3 main.py data/background.jpg data/portrait.jpg 1
```

7. 查看结果  
执行`main.py`文件后，可在工程目录`result`中查看背景替换结果。


8.其他说明


为了达到良好的背景替换效果，输入的人像jpg图片构图应尽可能简单，仅包含单个人像及其相应的背景，其中人像应与其他物体有一定的间隔并显示出完整的轮廓。
