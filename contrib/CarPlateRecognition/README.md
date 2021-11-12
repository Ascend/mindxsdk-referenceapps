# 车牌识别

## 1. 介绍

本样例是基于mxBase开发的端到端推理的C++应用程序，使用车牌检测模型和车牌识别模型在昇腾芯片上对图像中的车牌进行检测，并对检测到的图像中的每一个车牌进行识别，最后将可视化结果保存为图片形式。

对于车牌检测模型，在绝大多数情况下都能将车牌正确框选出来，车牌检测准确率较高；但受限于车牌识别模型的性能，只能识别蓝底车牌，对于黄底车牌和绿底新能源车牌很难正确识别，且模型对中文字符的识别准确率偏低，对于大角度车牌的识别准确率偏低。

本样例的主要处理流程为： Init > ReadImage > Resize > Detection_Inference > Detection_PostProcess > Crop_Resize1 >

Recognition_Inference > Recognition_PostProcess > WriteResult > DeInit。

### 1.1 支持产品

昇腾310(推理)

### 1.2 支持的版本

本样例配套的CANN版本为[1.77.22.6.220](https://www.hiascend.com/software/cann/commercial)，MindX SDK版本为[2.0.2](https://www.hiascend.com/software/mindx-sdk/mxvision)。

MindX SDK安装前准备可参考[《用户指南》](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)。

###  1.3 代码目录结构与说明

本样例工程名称为CarPlateRecognition，工程目录如下图所示：

```
├── include #头文件目录
  ├── carplate_recognition.h 
  ├── retinaface_postprocess.h 
  ├── lpr_postprocess.h 
  ├── initparam.h # 定义了包含程序所需参数的结构体
  ├── CvxText.h 
├── model #模型目录
  ├── lpr.aippconfig #转om模型所使用的aipp配置文件
  ├── retinaface.aippconfig 
  ├── pth2onnx.py #pth模型转onnx模型脚本
├── src #源文件目录
  ├── main.cpp #主程序
  ├── carplate_recognition.cpp #车牌识别流程处理函数源文件
  ├── retinaface_postprocess.cpp #车牌检测模型后处理源文件
  ├── lpr_postprocess.cpp #车牌识别模型后处理源文件
  ├── CvxText.cpp #定义了使用FreeType库在图片上写中文的类
├── imgs #README图片目录
├── build.sh
├── CMakeLists.txt
├── README.md
├── simhei.ttf # 黑体字体文件
```

### 1.4 技术实现流程图

![技术流程图](https://gitee.com/zhong-wanfu/mindxsdk-referenceapps/raw/master/contrib/CarPlateRecognition/imgs/技术流程图.jpg))

## 2. 环境依赖

环境依赖软件和版本如下表：

| 软件                | 版本                                                         | 说明                                               |
| ------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| mxVision            | [mxVision 2.0.2](https://www.hiascend.com/software/mindx-sdk/mxvision) | mxVision软件包                                     |
| Ascend-CANN-toolkit | [CANN 1.77.22.6.220](https://www.hiascend.com/software/cann/commercial) | Ascend-cann-toolkit开发套件包                      |
| 操作系统            | [Ubuntu 18.04](https://ubuntu.com/)                          | Linux操作系统                                      |
| OpenCV              | 4.3.0                                                        | 用于结果可视化                                     |
| FreeType            | [2.10.0](https://download.savannah.gnu.org/releases/freetype/) | 用于在图片上写中文(opencv只支持在图片上写英文字符) |

FreeType2.10.0的[安装教程](https://blog.csdn.net/u014337397/article/details/81115439)如下：

```
STEP1:从上面的FreeType版本链接中获取安装包freetype-2.10.0.tar.gz，保存到服务器
STEP2:进入freetype的安装目录：cd /home/xxx/freetype-2.10.0 # 该路径需用户根据实际情况自行替换
STEP3:执行配置命令：./configure --without-zlib
STEP4:执行编译命令：make
STEP5:执行安装命令：make install # 该步骤需要root权限，否则会提示安装失败
```

在进行模型转换和编译运行前，需设置如下的环境变量：

```shell
export MX_SDK_HOME=${MX_SDK_HOME}
export install_path=${install_path}
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH:.
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:${MX_SDK_HOME}/python:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export FREETYPE_HOME=${FREETYPE_HOME}
```

注：**${MX_SDK_HOME}** 替换为用户自己的MindX_SDK安装路径（例如："/home/xxx/MindX_SDK/mxVision"）；

​       **${install_path}** 替换为开发套件包所在路径（例如：/usr/local/Ascend/ascend-toolkit/latest）。

​       **${FREETYPE_HOME}** 需设置为用户自己的FreeType库的安装路径（例如：/home/xxx/freetype-2.10.0/include）。

### 3. 模型转换

模型转换使用的是ATC工具，具体使用教程可参考[《ATC工具使用指南》](https://support.huawei.com/enterprise/zh/doc/EDOC1100191944/a3cf4cee)。车牌检测模型和车牌识别模型转换所需的aipp配置文件均放置在/CarPlateRecognition/model目录下。

### 3.1 车牌检测模型的转换

**步骤1** **模型获取** 将[车牌检测项目原工程](https://hub.fastgit.org/zeusees/License-Plate-Detector/tree/master)克隆到**本地**。

```shell
git clone https://hub.fastgit.org/zeusees/License-Plate-Detector.git # 使用的镜像源
或
git clone https://github.com/zeusees/License-Plate-Detector.git
```

**步骤2** **pth转onnx** 将**pth2onnx.py**脚本放至**本地**工程目录下，执行如下命令：

```
python pth2onnx.py
```

若在模型转换过程中报错 “ **ValueError:Expected a cuda device,but got:cpu** ”，则说明是因为电脑上未安装cuda(cuda只适用于英伟达显卡)，可以将设备替换成CPU，方法如下：

```
将pth2onnx.py中的第10，11行代码：
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_dict = torch.load(weights, map_location=lambda storage, loc: storage.cuda(device))
替换为：
device = torch.device("cpu")
pretrained_dict = torch.load(weights, map_location=torch.device('cpu'))
```

*版本要求：*

*Python = 3.8.3*

*Pytorch = 1.7.0*

*onnx = 1.10.1*

*cuda = 10.2.120*

注：若原工程链接失效，可以直接下载已经转换好的[mnet_plate.onnx](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MMNET/model.zip)模型。

**步骤3** **onnx转om** 将步骤2中转换获得的onnx模型存放至**服务器端**的CarPlateRecognition/model/目录下，执行如下命令：

```shell
atc --model=./mnet_plate.onnx --output=./retinaface --framework=5 --soc_version=Ascend310 --input_format=NCHW --input_shape="image:1,3,640,640" --output_type=FP32 --insert_op_conf=./retinaface.aippconfig
```

### 3.2 车牌识别模型的转换

**步骤1** **模型获取** 下载车牌识别预训练模型的[lpr.prototxt](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MMNET/model.zip)文件和[lpr.caffemodel](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MMNET/model.zip)文件 。

**步骤2** **模型存放** 将获取到的.prototxt文件和.caffemodel文件存放至**服务器端**的CarPlateRecognition/model/目录下。

**步骤3** **caffe转om** 进入model文件夹，执行如下的ATC命令进行模型转换：

```shell
atc --model=./lpr.prototxt --weight=./lpr.caffemodel --output=./lpr --framework=0 --soc_version=Ascend310 --input_format=NCHW --input_shape="data:1,3,72,272" --output_type=FP32 --insert_op_conf=./lpr.aippconfig 
```

## 4. 编译与运行

**步骤1** **修改CMakeLists.txt文件** 

第**10**行 **set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")** 语句是设置MindX_SDK的安装路径，需将**$ENV{MX_SDK_HOME}**替换为用户实际的MindX_SDK安装路径。

第**12**行 **set(FREETYPE_HOME"$ENV{FREETYPE_HOME}")** 语句是设置FreeType库的安装路径，需将**$ENV{FREETYPE_HOME}**替换为用户实际的FreeType库安装路径。

第**39**行 **freetype** 语句是链接到FreeType库，该名称一般是不用修改的，若命名不同则看情况修改。

**步骤2** **编译**  执行shell脚本或linux命令对代码进行编译：

```shell
bash build.sh
或
rm -r bin # 删除原先的bin目录(如果有的话)
mkdir bin # 创建一个新的bin目录
rm -r build # 删除原先的build目录(如果有的话)
mkdir build # 创建一个新的build目录
cd build # 进入build目录
cmake .. # 执行cmake命令，在build下生成MakeFile文件
make # 执行make命令对代码进行编译
```

**步骤3** **推理** 请自行准备**jpg/jpeg**格式图像保存在工程目录下，执行如下命令：

```shell
./bin/car_plate_recognition ./xxx.jpeg # 自行替换图片名称
```

















