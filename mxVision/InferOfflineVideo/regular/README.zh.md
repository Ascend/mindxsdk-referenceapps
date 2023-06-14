

# InferOfflineVideo

## 1 简介

InferOfflineVideo基于mxVision SDK开发的参考用例，以昇腾Atlas310B卡为主要的硬件平台，用于在视频流中检测出目标。

## 2 环境依赖

- 支持的产品

本项目以昇腾Atlas 500 A2为主要的硬件平台。

- 软件依赖

| 软件名称 | 版本   |
| -------- | ------ |
| python    | 3.9.2     | 
| MindX SDK     |    5.0RC1    |
| CANN | 310使用6.3.RC1<br>310B使用6.2.RC1 |

## 3 准备

**步骤1：** 参考安装教程《mxVision 用户指南》安装 mxVision SDK。

**步骤2：** 配置环境变量。

```
. /usr/local/Ascend/ascend-toolkit/set_env.sh #toolkit默认安装路径，根据实际安装路径修改
. ${SDK_INSTALL_PATH}/mxVision/set_env.sh
```

**步骤3：** 转换模型
进入models目录，下载YOLOv3模型。[下载地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/ActionRecognition/ATC%20YOLOv3%28FP16%29%20from%20TensorFlow%20-%20Ascend310.zip)， 将下载的模型放入models文件夹中

执行转换命令
```
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310B1 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"
# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
```
执行完模型转换脚本后，会生成相应的.om模型文件。

**步骤4：** 修改regular/pipeline/regular.pipeline文件：

①：将所有“rtspUrl”字段值替换为可用的 rtsp 流源地址（目前只支持264格式的rtsp流，例："rtsp://xxx.xxx.xxx.xxx:xxx/input.264", 其中xxx.xxx.xxx.xxx:xxx为ip和端口号）；

②：将所有“deviceId”字段值替换为实际使用的device的id值，可用的 device id 值可以使用如下命令查看：`npu-smi info`

③：如需配置多路输入视频流，需要配置多个拉流、解码、缩放、推理、序列化插件，然后将多个序列化插件的结果输出发送到串流插件mxpi_parallel2serial（有关串流插件使用请参考《mxVision 用户指南》中“串流插件”章节），最后连接到appsink0插件。

## 4 运行

下载coco.names文件[链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/contrib/Collision/model/coco.names), 放在models目录下。

运行
`bash run.sh`

正常启动后，控制台会输出检测到各类目标的对应信息，结果日志将保存到`${安装路径}/mxVision/logs`中

手动执行ctrl + C结束程序

## 5 交叉编译

**步骤1：** 在Ubuntu18.0.4 x86_64系统上安装ARM版本的CANN套件包

**步骤2：** 在Ubuntu18.0.4 x86_64系统上安装ARM版本的SDK套件包

> 2.0.3及以下版本SDK不支持交叉编译，从2.0.4版本开始支持，安装的ARM版本的CANN包应与SDK版本适配，安装SDK套件包之前，确保环境中已安装x86版本的CANN包。

**步骤3：** 在开发环境执行如下命令检查是否安装，若已经安装则可以忽略

```
aarch64-linux-gnu-g++ --version
```

执行如下命令安装交叉编译工具链：

```
sudo apt-get install g++-aarch64-linux-gnu
```

**步骤4：** 下载zlib源码并编译

- 下载zlib tar.gz包：[下载地址](https://github.com/madler/zlib/releases/tag/v1.2.11)

- 解压并编译zlib源码：

  ```
  tar zxvf zlib-1.2.11.tar.gz
  ```

- 将build_zlib.sh脚本拷贝到同级目录下并执行如下命令进行编译；

  ```
  bash build_zlib.sh
  ```

- 编译完成后，将生成文件拷贝至sdk的opensource

  ```
  cp -r zlib/* ${MX_SDK_HOME}/opensource
  ```

**步骤5：** 交叉编译

- 修改目录下的CMakeLists.txt文件

  ```
  set(MX_SDK_HOME ${SDK安装路径})
  ...
  link_directories(
          ${MX_SDK_HOME}/lib
          ${MX_SDK_HOME}/opensource/lib
          ${MX_SDK_HOME}/lib/modelpostprocessors
          #arm版本cann的链接库路径
          /xxx/Ascend/ascend-toolkit/5.0.3/arm64-linux/runtime/lib64/stub
  )
  ```

- 修改目录下的build_x86.sh脚本

  ```
  export MX_SDK_HOME=${SDK安装路径}
  export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":"${arm版本cann包安装路径}/acllib/lib64":${LD_LIBRARY_PATH}
  ...
  CC=${aarch64-linux-gnu-gcc安装路径} CXX=${aarch64-linux-gnu-g++安装路径} cmake -S . -Build
  ```

- 执行脚本进行交叉编译，若生成可执行文件main，则说明编译成功

  ```
  bash build_x86.sh
  ```

**步骤6：** 运行

- 将编译成功的参考用例程序与安装成功的SDK文件全部打包，上传至A500服务器

- 解压上述两个压缩包

- 设置环境变量

  - 运行sdk目录下的set_env.sh脚本设置环境变量

    ```
    .  set_env.sh
    ```

  - A500环境中预安装有nnrt套件包，默认路径为/opt/ascend/nnrt，运行如下命令设置环境变量

    ```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ascend/nnrt/5.0.3/arm64-linux/runtime/lib64
    ```

- 进入参考样例目录下，执行 `./main` 运行程序

## FAQ

- 若在交叉编译或运行过程中，出现缺少gcc共享库的情况，可以从其他ARM框架下将该库拷贝至缺少的服务器中。

> 若是编译时缺少库，将缺少的库拷贝aarch64-linux-gnu目录下；
>
> 若是运行时缺少库，拷贝路径可参考原服务器中改库的存放路径

