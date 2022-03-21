# 2.1 图像检测sample样例

## 2.1.1 样例介绍

提供了一个图像检测sample样例，实现对本地图片进行YOLOv3目标检测，并把可视化结果保存到本地。
[样例获取](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/ImageDetectionSample)

## 2.1.2 运行前准备
参考[IDE开发环境搭建](./1-2IDE开发环境搭建.md)章节搭建好项目运行环境。
参考[Cmake介绍](./Cmake介绍.md)修改CMakeLists.txt文件。

### 2.1.2.1 模型转换
**步骤1** 在ModelZoo上下载YOLOv3模型 ，选择“历史版本”中版本1.1下载。[下载地址](https://www.hiascend.com/zh/software/modelzoo/detail/2/24a26134237f41a3974978d249451d19)

**步骤2** 将获取到的YOLOv3模型pb文件存放至："样例项目所在目录/model/"，如：  
![9.png](img/1623229532350.png '9.png')

**步骤3** 模型转换
具体模型转换步骤请参考C++样例目录下的README.md文件  
在步骤2目录中执行完模型转换脚本后，会生成相应的.om模型文件。

>模型转换使用了ATC工具，如需更多信息请参考:  https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html
### 2.1.2.2 配置pipeline
在test.pipeline文件中配置所需的模型路径与模型后处理插件路径。  
![10.png](img/1623231415247.png '10.png')  

![11.png](img/1623231423039.png '11.png')  

后插件路径根据SDK安装路径决定，一般情况下无需修改。
运行时如果出现找不到插件的报错，可以通过`find -name libyolov3postprocess.so`搜索找到路径后再更改pipeline中的值。  
![12.png](img/1623231850273.png '12.png')

### 2.1.2.3 配置Clion运行配置
参考[IDE开发环境搭建](./1-2IDE开发环境搭建.md)在Clion中添加环境变量。

点击Clion任务栏 Run->Edit Configurations->Working directory 填写当前工程目录位置。  
![13.png](img/1623232978995.png '13.png')

## 2.1.3 项目运行
完成前置步骤后，点击Run->Run"xxxx"（项目名称）运行项目。

成功运行项目后，程序会将检测结果保存在远程环境项目目录下result.jpg中。


右击工程 Deployment->Download from...，将远程生成的结果文件下载到本地中，同步两端文件，然后用户就可以在本地中查看项目运行结果了。
本样例中输出结果为在输入的test.jpg中，将可识别的对象画框并显示识别类型后输出的result.jpg  
![4.png](img/1623382648767.png '4.png')

![14.png](img/1623382869487.png '14.png')


## 2.2.1 Python样例介绍
提供了一个图像检测sample样例，实现对本地图片进行YOLOv3目标检测，并把可视化结果保存到本地。

## 2.2.2 运行前准备
请参考[IDE开发环境搭建](./1-2IDE开发环境搭建.md)章节搭建好项目运行环境。
将项目从（[项目文件地址](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/ImageDetectionSample/python)）移动到运行项目的目录下。

### 2.2.2.1 模型转换
确认*.om模型文件存在model 路径下。
如果不存在则需要执行模型转换步骤，请参考本章节C++ 样例运行 模型转换步骤。

### 2.2.2.2 配置pipeline
python pipeline在脚本main.py内部，其中模型路径和工程中model文件夹路径和SDK安装路径相匹配的，一般情况下不需要修改。
**注意**将pipeline中红框部分的${SDK安装路径}替换为自己的SDK安装路径。  
![image.png](img/20210712150707.png 'image.png')  
如果有修改路径需求可以通过`find -name libyolov3postprocess.so`相关命令搜索到需要的依赖文件，更改pipeline中的值。  
![12.png](img/1623231850273.png '12.png')
### 2.2.2.3 模型转换
模型转换请参考本章节C++样例运行模型转换内容。


### 2.2.2.4 配置pyCharm运行配置
参考[IDE开发环境搭建](./1-2IDE开发环境搭建.md)在pyCharm中添加环境变量。

点击pyCharm任务栏 Run->Edit Configurations->Working directory 填写当前工程目录位置。  
![image.png](img/1623389741249.png 'image.png')

## 2.2.3 项目运行
完成前置步骤后，准备一张待检测图片保存为test.jpg放入项目和main.py同级的文件夹中，点击Run->Run"mian" 运行项目。

成功运行项目后，程序会将检测结果保存在远程环境项目目录下result.jpg中。

右击工程 Deployment->Download from...，将远程生成的结果文件下载到本地中，同步两端文件，然后用户就可以在本地中查看项目运行结果了。
本样例中输出结果为在输入的test.jpg中，将可识别的对象画框并显示识别类型后输出的result.jpg  

![image.png](img/1623835106290.png 'image.png')
## 2.3 yolov3模型转换脚本
以下为yolov3模型转换脚本示例，使用时请确认参数中的路径是实际的相关路径。
```
#!/bin/bash

# 该脚本用来将pb模型文件转换成.om模型文件
# This is used to convert pb model file to .om model file.


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
