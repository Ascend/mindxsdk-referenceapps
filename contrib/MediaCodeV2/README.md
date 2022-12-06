# 基于Mxbase V2接口的Media Codec

## 1 介绍

基于MindX SDK mxVision 3.0RC3 开发Media Codec的程序。本程序采用c++开发，使用MxBase V2接口。通过FFmpeg的拉流操作对输入的视频数据进行处理，将拉流后的帧数据进行视频解码处理为YUV格式图片，并将处理过的帧图片进行缩放到要求的尺寸。然后将缩放过的帧图片进行视频编码处理。最后，将编码后得到的数据写入文件，可以与原视频进行对比。注意需要自行准备视频进行验证。视频转码不止可以进行一路的视频转码，还可以进行多路视频转码，提高转码效率。

程序输入：任意.h264格式或者.264格式视频

程序输出：输出经过缩放的.h264格式或者.264格式的视频

### 1.1 模型介绍

视频转码是实现将视频解码、缩放、编码的流程。视频编码又称为视频压缩，伴随着用户对高清视频的需求量的增加，视频多媒体的视频数据量也在不断加大。如果未经压缩，这些视频很难应用于实际的存储和传输。而视频中是有很多冗余信息的，以记录数字视频的YUV分量格式为例，YUV分别代表亮度与两个色差信号。以4：2：2的采样频率为例，Y信号采用13.5MHz，色度信号U和V采用6.75MHz采样，采样信号以8bit量化，则可以计算出数字视频的码率为：13.5*8 + 6.75*8 + 6.75*8= 216Mbit/s。如此大的数据量如果直接进行存储或传输将会遇到很大困难，因此必须采用压缩技术以减少码率。

参考地址：

https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/mxVision/MediaCodec/Ascend310

### 1.2 支持的产品

昇腾310(推理)

### 1.3 支持的版本

本样例配套的CANN版本为 [5.1.RC1](https://www.hiascend.com/software/cann/commercial) ，MindX SDK版本为 [3.0RC3](https://www.hiascend.com/software/Mindx-sdk) 。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.4 目录结构

```
.
|
|-------- logs                                      // 存放视频转码的log信息   
|           |---- .keep                  
|-------- mxbase
|           |---- CMakeLists.txt
|           |---- MediaCodecV2.cpp                  // 视频转码的源文件
|           |---- MediaCodecV2.h                    // 视频转码的头文件
|           |---- build.sh                          // 编译脚本
|           |---- MainV2.h             
|           |---- run.sh                            // 运行多路视频转码的脚本
|           |---- show.sh                           // 运行显示log信息的脚本
|           |---- stop.sh                           // 停止多路视频转码的脚本
|-------- out                                       // 存放输出结果 
|           |---- .keep 
|-------- test
|           |---- .keep                             // 测试视频(需自行准备)
|-------- README.md

```

### 1.5 特性及适用场景

MediaCodecV2是基于v2接口的视频转码，适用于.h264格式或者.264格式的视频进行视频转码，
在帧率为25fps的视频上，MediaCodecV2的性能和精度可以达到和v1接口一致，但是在其他的情况下的效果不够理想。
视频每秒的转码帧率只能达到25fps,当视频帧率不满足25fps,结果不能达到。

## 2 环境依赖 

### 2.1 软件版本

| 软件                | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 3.0RC3       | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| Ascend-CANN-toolkit | 5.1.RC1      | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |
| 操作系统           | Ubuntu 18.04  | 操作系统                    | Ubuntu官网获取      | 
| FFmpeg            | FFmpeg 4.2.1  | 视频音频处理工具包           | [链接](https://github.com/FFmpeg/FFmpeg/archive/n4.2.1.tar.gz)     |

### 2.2 安装FFmpeg

下载完解压，按以下命令编译即可
```
./configure --prefix=/usr/local/ffmpeg --enable-shared
make -j
make install

```

### 2.3 配置环境变量

```
# Mind SDK环境变量:
.${SDK-path}/set_env.sh

# CANN环境变量:
.${ascend-toolkit-path}/set_env.sh

# FFmpeg环境变量:
export LD_LIBRARY_PATH=${FFmpeg安装路径}/lib:$LD_LIBRARY_PATH

# 环境变量介绍
SDK-path:SDK mxVision安装路径
ascend-toolkit-path:CANN安装路径
env

```
`mxbase`文件夹中的`CMakeLists.txt`文件中涉及到上面相关的环境变量，也需要在文件中进行相应的配置。

## 3 V2接口运行

1) 准备一个测试视频，置于 test 文件夹中（仅支持.h64格式或者.264格式的视频，且视频帧率为25fps）

2) 进入工程目录

3) 代码编译：参考`mxbase/build.sh`脚本，将c++代码进行编译

进入`mxbase`目录，键入执行指令，编译代码：

```c++
bash build.sh
```
代码编译成功会在`mxbase`目录下，生成可执行文件`mediacodecV2`

注意：代码中使用的BlockingQueue类是使用的开源代码，需自行添加

代码所在位置：https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/mxBaseVideoSample/BlockingQueue

4) 输入执行指令，发起视频转码性能测试：
```c++
./mediacodecV2 ${测试视频路径} ${输出结果路径}
例如: ./mediacodecV2 ../test/test1.264 ../out/out.264
```
执行完毕后，会将程序输出的视频转码的结果，保存在工程目录下`out`中 (在运行命令前保证`out`存在，否则会影响程序的运行）


## 4 脚本运行

## 4.1 运行多路

通过计算视频编码的帧率和原视频的帧率进行对比。

1）准备测试视频：自行准备.h264格式或者.264格式的视频，且视频帧率为25fps。

2）进入`mxbase`目录

3）在`run.sh `脚本，可修改测试视频的路径

```
    nohup ./mediacodecV2 ${test_path}/xxx.264 ${out_path}/output${i}.264 > ${log_path}/output${i}.log 2>&1 &
    //xxx.264为自行准备的测试视频
    // output${i}.264 后缀名根据准备的测试视频进行修改，.h264或者.264。
```

4）运行`run.sh `脚本，得到测试视频的输出和log信息

键入执行指令，对测试视频发起多路转码验证测试：

```c++
bash run.sh ${运行路数}
例如: bash run.sh 5
```
执行完毕后，会在`out`和`logs`文件夹输出处理过的视频和log文件（转码时间在一定范围内浮动）

## 4.2 log信息显示

1）进入`mxbase`目录

2）运行`show.sh `脚本，得到测试视频的log信息

注意：运行多路视频转码后，需等待一段时间，才能显示完整的log信息，否则只显示每秒的编码帧率，因为视频转码会根据不同的转码视频的大小运行不同的时间。

键入执行指令，展示log文件信息：

```c++
bash show.sh
```

执行完毕后，会在控制台输出frame的resize大小、编码帧率和推理时间（帧率和推理时间在一定范围内浮动）。

展示的log信息如下所示：

```
I20221202 15:32:14.838917 18851 MediaCodecV2.cpp:193] ReszieWidth = 352, ResizeHight = 288
I20221202 15:32:12.035985 18854 MediaCodecV2.cpp:380] video encode frame rate for per second: 25 fps.
I20221202 15:32:13.036136 18854 MediaCodecV2.cpp:380] video encode frame rate for per second: 25 fps.
I20221202 15:32:14.036370 18854 MediaCodecV2.cpp:380] video encode frame rate for per second: 26 fps.
I20221202 15:32:15.037037 18854 MediaCodecV2.cpp:380] video encode frame rate for per second: 21 fps.
I20221202 15:32:15.117311 18742 MediaCodecV2.cpp:447] Total decode frame rate: 25.3556 fps.
I20221202 15:32:15.117194 18742 MediaCodecV2.cpp:445] total process time: 205.122s.
```

## 4.3 停止多路视频转码
1）进入`mxbase`目录

2）运行`stop.sh `脚本

键入执行指令，停止多路视频转码：

```c++
bash stop.sh
```

执行完毕后，会在控制台显示停止多路转码的情况。

## 5 精度+性能验证

## 5.1 V2精度+性能验证

视频转码的路数和resize的宽高有关系，所以为了保证多路转码过程中，编码帧率保持稳定，所以在进行多路转码的时候，需要修改`MediaCodecV2.cpp`中的`RESIZE_WIDTH`和`RESIZE_HEIGHT`。

运行10路视频转码

步骤1：在`MediaCodecV2.cpp`中分别修改`RESIZE_WIDTH`和`RESIZE_HEIGHT`为`720`和`480`，将`test1.264`的视频转码为`D1`格式的视频。

```
    const uint32_t RESIZE_WIDTH = 720;
    const uint32_t RESIZE_HEIGHT = 480;
```

步骤2：进入`mxbase`目录，编译脚本。

步骤3：运行多路视频转码。

键入执行指令，对test1.264的测试视频发起精度及性能验证测试：

```c++
bash build.sh

bash run.sh ${运行路数}
例如: bash run.sh 10

bash show.sh  //显示log结果
```
执行完毕后，会在控制台输出10路log信息，显示frame的resize大小、编码帧率和推理时间（帧率和推理时间在一定范围内浮动）。

## 5.2 验证结论

对v1和v2运行结果进行对比，得出精度及性能验证结论：

v1的结果如下：

```
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder0) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder0) fps (25).
I20221204 00:27:32.643878  8719 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder1) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder1) fps (25).
I20221204 00:27:32.682122  8744 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder2) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder2) fps (25).
I20221204 00:27:32.755513  8780 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder3) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder3) fps (25).
I20221204 00:27:32.858893  8815 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder4) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder4) fps (25).
I20221204 00:27:32.881512  8826 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder5) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder5) fps (25).
I20221204 00:27:32.957969  8861 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder6) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder6) fps (25).
I20221204 00:27:33.022509  8884 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder7) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder7) fps (25).
I20221204 00:27:33.139667  8925 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder8) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder8) fps (25).
I20221204 00:27:33.151942  8932 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder9) fps (25).
I20221204 00:27:32.590896  8697 MxpiVideoEncoder.cpp:352] Plugin(mxpi_videoencoder9) fps (25).
    
```

v2的结果如下：
```
I20221205 16:05:10.194156  2537 MediaCodecV2.cpp:193] ReszieWidth = 720, ResizeHight = 480
I20221205 16:05:06.895665  2540 MediaCodecV2.cpp:380] video encode frame rate for per second: 26 fps.
I20221205 16:05:07.895879  2540 MediaCodecV2.cpp:380] video encode frame rate for per second: 25 fps.
I20221205 16:05:08.896096  2540 MediaCodecV2.cpp:380] video encode frame rate for per second: 25 fps.
I20221205 16:05:09.896323  2540 MediaCodecV2.cpp:380] video encode frame rate for per second: 26 fps.
I20221205 16:05:10.896548  2540 MediaCodecV2.cpp:380] video encode frame rate for per second: 8 fps.
I20221205 16:05:10.896868  2517 MediaCodecV2.cpp:447] Total decode frame rate: 25.2418 fps.
I20221205 16:05:10.896737  2517 MediaCodecV2.cpp:445] total process time: 206.047s.
```

精度：V2接口和v1接口一样，根据不同转码格式，运行相同路数的视频转码，输出转码视频的格式分辨率和转码格式的分辨率保持一致，精度达标。

性能：原视频帧率为25fps,多路视频转码过程中，每秒的编码帧率一直保持在25fps左右，平均帧率均在25.2fps左右，转码前后帧率均为25fps，性能达标。

结论：V2接口的性能和精度达标。

注意：log信息最后一秒的帧率可能不满足25fp,但是不影响性能和结果。

## 6 常见问题

### 6.1 路径问题

**问题描述：**
```
提示：Couldn't open input stream ../test/test.264.
```

**解决方案：**

输入的视频不存在，检查输入路径是否正确。


### 6.2 输出问题

### 6.2.1 运行命令前没有输出的out文件夹
**问题描述：**
```
提示：failed to open file.
```

**解决方案：**

运行命令前新建out文件夹。


### 6.2.2 运行命令没有输入输出路径
**问题描述：**
```
提示：please input output path, such as ../out/out_test.h264.
```

**解决方案：**

运行命令时输入输出路径。


### 6.3 格式问题

**问题描述：**
```
提示：Couldn't decode mp4 file.
```

**解决方案：**

将输入视频更换为.h264格式或者.264格式视频。


### 6.4 环境配置问题

**问题描述：**
```
提示类似：error while loading shared libraries.so.3:cannot open shared object file.
```

**解决方案：**

在安装好FFmpeg之后，导入相关的环境变量。
```
vim ~/.bashrc
export MX_SDK_HOME=${SDK安装路径}
export LD_LIBRARY_PATH=${FFmpeg安装路径}/lib:$LD_LIBRARY_PATH

```