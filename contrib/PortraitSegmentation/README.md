# MindXSDK 人像分割与背景替换

## 1 简介
  本开发样例基于MindX SDK实现了端到端的人像分割与背景替换（Portrait Segmentation and Background Replacement, PSBR）。PSBR的主要功能是使用Portrait模型对输入图片中的人像进行分割，然后与背景图像融合，实现背景替换。  
样例输入：带有人像的jpg图片和一张背景图像。  
样例输出：背景替换后的图像。<br/>
## 2 目录结构
本工程名称为PortraitSegmentation，工程目录如下图所示：
```
|-------- data                                // 存放测试背景及人像图片
|-------- models
|           |---- portrait.pb                 // 人像分割pb模型
|           |---- insert_op.cfg               // 模型转换配置文件
|           |---- portrait.om                 // 人像分割om模型
|-------- pipline
|           |---- segment.pipeline            // 人像分割模型流水线配置文件
|-------- result                              // 存放测试结果
|-------- main.py                             
|-------- README.md   
```
## 3 依赖

| 软件名称 | 版本   |
| :--------: | :------: |
|ubantu 18.04|18.04.1 LTS   |
|MindX SDK|2.0.2|
|Python| 3.7.5|
|numpy | 1.18.2 |
|opencv_python|3.4.0|

请注意MindX SDK使用python版本为3.7.5，如出现无法找到python对应lib库请在root下安装python3.7开发库  
`apt-get install libpython3.7`
## 4 模型转换
人体语义分割采用提供的Portrait.pb模型。由于原模型是基于tensorflow的人像分割模型，因此我们需要借助于ATC工具将其转化为对应的om模型。  
**步骤1**  在ModelZoo上下载Portrait原始模型：[下载地址](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/PortraitNet%20/portrait.pb)      
&ensp;&ensp;&ensp;&ensp;&ensp; 对应的cfg文件：[下载地址](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/PortraitNet%20/insert_op.cfg)  

**步骤2**  将获取到的Portrait模型pb文件和cfg文件存放至：“项目所在目录/models”  

**步骤3**  模型转换  

在pb文件所在目录下执行以下命令  
```
#设置环境变量（请确认install_path路径是否正确）  
#Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest    

export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH 
 
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

https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html

## 5 测试

1. 获取om模型   
```
见4： 模型转换
```
2. 配置[环境准备和依赖安装]（https://gitee.com/ascend/samples/tree/master/python/environment） 
```   
#执行如下命令，打开.bashrc文件
cd $home
vi .bashrc
#在.bashrc文件中添加以下环境变量:

export MX_SDK_HOME=${SDK安装路径}/mxVision

export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64` 

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
进入工程文件的data目录下，下载对应jpg格式的测试图片，并分别命名为background.jpg以及portrait.jpg。

5. 运行可执行文件
```

切换至工程主目录，执行以下命令运行样例。命令行格式为 [python3.7 main.py 背景图片路径 人像图片路径]

例：python3.7 main.py data/background.jpg data/portrait.jpg
```

6. 查看结果  
执行`main.py`文件后，可在工程目录`result`中查看背景替换结果。
