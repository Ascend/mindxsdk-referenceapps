# FFmpeg-Ascend

## 介绍

MxVison ascend 硬件平台内置了视频相关的硬件加速解码器，
为了用户的易用性，MxVision 提供了 FFmepg-Ascend 解决方案。

该样例的处理流程为：

```
准备芯片及环境 > 安装 CANN 版本包 > 拉取 FFmpeg-Ascend 代码 > 编译 > 执行
```

## 支持的产品
Atlas 300I Pro

## 支持的 ACL 版本

5.1.RC2
*注意：目前只支持 HiMpi 接口即 310P 昇腾 AI芯片。

查询 ACL 版本号的方法是，在 Atlas 产品环境下，运行以下命令：
```bash
npu-smi info
```

## 支持的功能
|功能|mpeg4|h264/h265|多路|
|:----:|:----:|:----:|:----:|
|硬件解码|√|√|√|
|硬件编码|√|√|√|
|硬件转码|√|√|√|
|硬件缩放|√|√|√|

## 安装 CANN 版本包
详情请参考: https://support.huawei.com/enterprise/zh/doc/EDOC1100234042/f1fad1e0

## FFmpeg-Ascend目录结构

FFmpeg-Ascend 目录主要文件：
.
|-- configure (FFmpeg 编译入口)
|
|-- README.md (FFmpeg-Ascend 使用手册)
|
|-- ffbuild (编译相关文件)
|
|-- fftools (FFmpeg 相关命令行工具)
|
|-- libavcodec (FFmpeg 编解码器)
|
|-- libavutil (FFmpeg 相关工具，包含硬件加速相关流程)

## 依赖条件

设置环境变量：
* `ASCEND_HOME`     Ascend 安装的路径，一般为 `/usr/local/Ascend`
* `FFMPEG_LIB_PATH` FFmpeg 编译安装的 lib 文件路径（一般为 FFmpeg 安装目录下的 lib 目录， 由安装时的 --prefix 编译选项来指定）。
* `LD_LIBRARY_PATH` 指定 ffmpeg 程序运行时依赖的动态库查找路径。

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=${FFMPEG_LIB_PATH}:${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## 编译

编写编译脚本
```bash
vi run.sh
```

`run.sh` 内编译选项: 
* `prefix` : FFmpeg 及相关组件安装目录
* `enable-shared` : FFmpeg 允许生成 so 文件
* `extra-cflags` : 添加第三方头文件
* `extra-libs` : 添加第三方 so 文件
* `enable-ascend` : 允许使用 ascend 进行硬件加速
```bash
./configure \
    --prefix=./ascend \
    --enable-shared \
    --extra-cflags="-I${ASCEND_HOME}/ascend-toolkit/latest/acllib/include"
    --extra-libs="-lacl_dvpp_mpi -lascendcl" \
    --enable-ascend \
    && make -j && make install
```

脚本添加可执行权限
```bash
chmod +x run.sh
```

运行脚本
```bash
./run.sh
```


注意：若是运行失败，则可能是 `./configure` 没有可执行权限
```bash
chmod +x ./configure
chmod +x ./ffbuild/*.sh
```

## 运行
当前目录或者 FFmpeg 安装目录下均会生成 `ffmpeg` 可执行文件，均可以使用。
相关指令参数：

* `-hwaccel`    -   指定采用 ascend 来进行硬件加速, 用来做硬件相关初始化工作。

解码相关参数(注意：解码相关参数需要在 `-i` 参数前设置)：
* `-c:v`        -   指定解码器为 h264_ascend (解码 h265 格式可以使用 h265_ascend)。
* `-device_id`  -   指定硬件设备 id 为 0。取值范围取决于芯片个数，默认为 0。 `npu-smi info` 命令可以查看芯片个数
* `-channel_id` -   指定解码通道 id [0-255], 默认为0, 若是指定的通道已被占用, 则自动寻找并申请新的通道。
* `-resize`     -   指定缩放大小, 输入格式为: {width}x{height}。宽高:[128x128-4096x4096], 宽高相乘不能超过 4096*2304。宽要与 16 对齐，高要与 2 对齐。 
* `-i`          -   指定输入文件（或者可以是 rtsp 流）。

编码相关参数(注意：解码相关参数需要在 `-i` 参数后设置)：
* `-c:v`        -   指定编码器为 h264_ascend (编码成 h265 格式可以使用 h265_ascend)。
* `-device_id`  -   指定硬件设备 id 为 0。取值范围取决于芯片个数，默认为 0。 `npu-smi info` 命令可以查看芯片个数。
* `-channel_id` -   指定编码通道 id [0-127], 默认为 0, 若是指定的通道已被占用, 则自动寻找并申请新的通道。
* `-profile`    -   指定视频编码的画质级别（0: baseline, 1: main, 2: high, 默认为 1。 H265 编码器只支持 main）。
* `-rc_mode`    -   指定视频编码器的速率控制模式（0: CBR, 1: VBR, 默认为 0）。
* `-gop`        -   指定关键帧间隔, [1, 65536], 默认为 30。
* `-frame_rate` -   指定帧率, [1, 240], 默认为25。
* `-max_bit_rate` - 限制码流的最大比特率, [2， 614400], 默认为 20000。
* `-movement_scene` - 指定视频场景（0：静态场景（监控视频等）， 1：动态场景（直播，游戏等））, 默认为 1。

```bash
./ffmpeg -hwaccel ascend -c:v h264_ascend -i test.264 -c:v h264_ascend out.264
```

```bash
./ffmpeg -hwaccel ascend -c:v h264_ascend -device_id 0 -channel_id 0 -resize 1024x1000 -i test.264 -c:v h264_ascend -device_id 0 -channel_id 0 -profile 2 -rc_mode 0 -gop 30 -frame_rate 25 -max_bit_rate 20000 out.264
```

```bash
./ffmpeg -hwaccel ascend -c:v h264_ascend -i test.264 out.yuv
./ffmpeg -hwaccel ascend -s 1920x1080 -pix_fmt nv12 -i out.yuv -c:v h264_ascend out.264
```