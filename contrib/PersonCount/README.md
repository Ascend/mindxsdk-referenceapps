人群密度计数

## 1 介绍
（项目的概述，包含的功能）
（项目的主要流程）
项目的概述：基于MindX SDK，在昇腾平台上，开发端到端人群计数-人群密度估计，输入一幅人群图像，输出图像当中人的计数（估计）的结果。  
  
项目的主要流程：  
（1）输入类型是图片数据（jpg图片序列）,本项目使用的数据集图片来源是https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PersonCount/data.zip   
（2）通过调用MindX SDK提供的图像解码接口mxpi_imagedecoder，解码后获取图像数据。  
（3）然后进行图像尺寸大小变换，调用MindX SDK提供的图像尺寸大小变换接口mxpi_imageresize插件，检测模式的输入图像大小要求高800，宽1408。  
（4）将尺寸变换后的图像数据输入人群计数模型进行推理,推理使用的caffemodel模型和om模型来源是https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/PersonCount/model.zip。  
（5）模型后处理，调用MindX SDK提供的模型推理插件mxpi_modelinfer，后处理配置文件存放的地址是"models/insert_op.cfg"。然后调用MindX SDK提供的插件mxpi_dataserialize，将stream结果组装成json字符串输出。  
（6）模型输出经过后处理后，得到人群密度估计图和人群计数估计值。  
场景限制：  
  输入的图像应为人群图像，图像中含有多个人，能取得较好的推理效果。  
### 1.1 支持的产品

可列出项目所用的硬件平台、支持的硬件平台、访问方式等  
人群计数项目使用的硬件平台为华为海思Ascend310。其中，om模型适配海思Ascend310硬件平台，模型的推理过程也由硬件平台Ascend310完成。  
  
### 1.2 支持的版本  
  
支持的SDK版本：20.2.0  
  

### 1.3 软件方案介绍

请先总体介绍项目的方案架构。如果项目设计方案中涉及子系统，请详细描述各子系统功能。如果设计了不同的功能模块，则请详细描述各模块功能。
项目的方案架构：  

我们将人群计数任务划分为多个子任务，并针对每个子任务设计相应的子系统来实现相应的功能。  
表1.1 系统方案各子系统功能描述：  

| 序号 | 子系统 | 功能描述     |  
| ---- | ------ | ------------ |  
| 1    | 模型转换    | 利用昇腾SDK提供的ATC转换工具将caffemodel转换成om模型|  
| 2    | 后处理插件 | 获取推理结果，并计算人数，图片数据归一化，ObjectInfo中的mask用来存归一化后的图片数据，ObjectInfo中的classId用来存估计的人数 |  

### 1.4 代码目录结构与说明

本工程名称为 [南开大学]人群密度统计，工程目录如下图所示：  
│  build.sh  //用于生成后处理共享库的编译命令使用sh build.sh完成编译  
│  main.py   //对待检测图片进行人群计数的主体程序，包含读入图片数据、模型推理、写热度图等功能特性  
│  run.sh    //人群计数的运行脚本，运行main.py文件  
├─accuracy and performance //存放精度与性能测试的代码   
│  └─test.py  //精度与性能测试的代码   
│
├─config     //配置文件夹  
│  |--insert_op.cfg //生成om模型的config文件  
│  └─person.names  //label文件  
│      
├─img  //存放readme使用到的png图片  
│  |--tech_arch.png  //1.5节使用的技术实现流程图  
|  └─err1.png       //章节6使用到的错误报告截图  
│        
├─model //转换后的om模型  
│  └─count_person_8.caffe.om  
│  
├─model transformation script
│  |--insert_op.cfg  //模型转换需要的配置文件
│  └─ model_conversion.sh //模型转换脚本
│ 
├─pipeline //本项目使用的前端是python开发，用到的pipeline配置嵌入到main.py，所以该文件夹为空  
├─Plugin1  //编译后处理插件所需的源文件，生成的共享库文件存放于build文件夹  
│  |--CMakeLists.txt  
│  |--Plugin1.cpp  
└─ └─Plugin1.h  

### 1.5 技术实现流程图

（可放入流程图片）  
![Image text](https://gitee.com/superman418/mindxsdk-referenceapps/raw/master/contrib/PersonCount/img/tech_arch.png)

## 2 环境依赖

请列出环境依赖软件和版本。

推荐系统为ubantu 18.04。

| 软件名称 | 版本   |
| -------- | ------ |
| MindX SDK mxManufacture    |    2.0.2    |
| ascend-toolkit             |    3.3.0    |

在编译运行项目前，需要设置环境变量：
MX_SDK_HOME="~/mxManufacture"  
LD_LIBRARY_PATH=\\${MX_SDK_HOME}/lib:\\${MX_SDK_HOME}/opensource/lib:\\${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit:\\${LD_LIBRARY_PATH}  
PYTHONPATH=\\${MX_SDK_HOME}/python:\\${PYTHONPATH}  

install_path=/usr/local/Ascend/ascend-toolkit/latest
PATH=/usr/local/python3.9.2/bin:\\${install_path}/atc/ccec_compiler/bin:\\${install_path}/atc/bin:\\$PATH  
PYTHONPATH=\\${install_path}/atc/python/site-packages:\\${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:\\${install_path}/atc/python/site-packages/schedule_search.egg  
LD_LIBRARY_PATH=\\${install_path}/atc/lib64:\\$LD_LIBRARY_PATH  
ASCEND_OPP_PATH=\\${install_path}/opp   

- 环境变量介绍  
MX_SDK_HOME指明MindX SDK mxManufacture的根安装路径，用于包含MindX SDK提供的所有库和头文件。  
LD_LIBRARY_PATH提供了MindX SDK已开发的插件和相关的库信息。  
install_path指明ascend-toolkit的安装路径。  
PATH变量中添加了python的执行路径和atc转换工具的执行路径。  
LD_LIBRARY_PATH添加了ascend-toolkit和MindX SDK提供的库目录路径。  
ASCEND_OPP_PATH指明atc转换工具需要的目录。  

具体执行命令  
export MX_SDK_HOME="~/mxManufacture"  
export LD_LIBRARY_PATH=\\${MX_SDK_HOME}/lib:\\${MX_SDK_HOME}/opensource/lib:\\${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit:\\${LD_LIBRARY_PATH}  

export install_path=/usr/local/Ascend/ascend-toolkit/latest  
export PATH=/usr/local/python3.9.2/bin:\\${install_path}/atc/ccec_compiler/bin:\\${install_path}/atc/bin:\\$PATH  
export PYTHONPATH=\\${install_path}/atc/python/site-packages:\\${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:\\${install_path}/atc/python/site-packages/schedule_search.egg  
export LD_LIBRARY_PATH=\\${install_path}/atc/lib64:\\$LD_LIBRARY_PATH  
export ASCEND_OPP_PATH=\\${install_path}/opp    


## 依赖安装

安装MindX SDK mxManufacture：
chmod u+x Ascend-mindxsdk-mxmanufacture_2.0.2_linux-aarch64.run
./Ascend-mindxsdk-mxmanufacture_2.0.2_linux-aarch64.run

ascend-toolkit:
从/usr/local/Ascend/ascend-toolkit/latest/路径获取

## 编译与运行
（描述项目安装运行的全部步骤，，如果不涉及个人路径，请直接列出具体执行命令）

示例步骤如下：
**步骤1** （修改相应文件）修改Plugin1目录下的CMakeLists.txt中PROJECT_SOURCE_DIR变量，该变量指向MindX SDK mxManufacture的根安装路径。修改main.py中的Dataset_Path变量，该变量指向待检测的图片路径。修改main.py中gt_num变量，改变了指向待检测图片的groundtruth.此外，如果想要得到pipeline中各个插件的具体运行时间，
可以修改mxManufacture SDK的sdk.conf文件，使得enable_ps变量为true.

**步骤2** （设置环境变量）按照第二章节设置环境变量所需的具体执行指令执行即可。

**步骤3** （执行编译的步骤）首先通过运行build.sh脚本文件生成后处理使用的共享库如sh build.sh。然后使用模型转换命令将caffe模型转化为om模型待使用，具体命令为atc --input_shape="blob1:8,3,800,1408" --weight="model/count_person.caffe.caffemodel" --input_format=NCHW --output="model/count_person_8.caffe" --soc_version=Ascend310 --insert_op_conf=model/insert_op.cfg --framework=0 --model="model/count_person.caffe.prototxt"。此外，我们已经在文件夹model transformation script提供了模型转换脚本将脚本文件复制到主目录即可运行.

**步骤4** （运行及输出结果）直接运行run.sh即可，生成的热度图保存在当前目录的heat_map文件夹下，并且每张热度图的命名以原图片名称为前缀以heatmap为后缀。此外，我们还在文件夹accuracy and performance code提供了精度与性能测试代码，将test.py和test.sh拷贝到主目录中然后执行sh test.sh即可运行精度与性能测试代码。




## 5 软件依赖说明

如果涉及第三方软件依赖，请详细列出。

| 依赖软件 | 版本  | 说明                     |
| -------- | ----- | ------------------------ |
| cmake     | 3.10.2 | 用于编译并生成后处理插件 |
| python    | 3.9.2     |  用于编译用户程序如main.py   |



## 6 常见问题

请按照问题重要程度，详细列出可能要到的问题，和解决方法。

### 6.1 batch问题

**问题描述：**
在使用batch机制时，模型需要的数据维度和输入到流中的数据维度不匹配。该问题造成的主要原因是因为老版的MindX_SDK不提供自动组batch功能。程序中使用多次senddata函数，每次send一张图片到流中，并假设tensorinfer插件能够自动组batch然后进行batch模型的推理。
截图或报错信息：
![err information](https://gitee.com/superman418/mindxsdk-referenceapps/raw/master/contrib/PersonCount/img/err1.png)
**解决方案：**

新版Ascend-mindxsdk-mxmanufacture_2.0.2_linux-aarch64中的tensorinfer插件能够自动组batch然后进行batch模型的推理，可以解决该问题。