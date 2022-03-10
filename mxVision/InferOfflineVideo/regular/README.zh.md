

# InferOfflineVideo

## 1 简介

InferOfflineVideo基于mxVision SDK开发的参考用例，以昇腾Atlas300卡为主要的硬件平台，用于在视频流中检测出目标。

## 2 环境依赖

- 支持的硬件形态和操作系统版本

| 硬件形态                             | 操作系统版本   |
| ----------------------------------- | -------------- |
| x86_64+Atlas 300I 推理卡（型号3010） | Ubuntu 18.04.1 |
| x86_64+Atlas 300I 推理卡（型号3010） | CentOS 7.6     |
| ARM+Atlas 300I 推理卡 （型号3000）   | Ubuntu 18.04.1 |
| ARM+Atlas 300I 推理卡 （型号3000）   | CentOS 7.6     |

- 软件依赖

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.5.1+ |
| mxVision | 0.2    |

## 3 准备

**步骤1：** 参考安装教程《mxVision 用户指南》安装 mxVision SDK。

**步骤2：** 配置 mxVision SDK 环境变量。

`export MX_SDK_HOME=${安装路径}/mxVision `

注：本例中mxVision SDK安装路径为 /root/MindX_SDK。

**步骤3：** 在regular目录下创建目录models `mkdir models`， 根据《mxVision 用户指南》中“模型支持列表”章节获取Yolov3种类模型，并放到该目录下。

**步骤4：** 修改regular/pipeline/regular.pipeline文件：

①：将所有“rtspUrl”字段值替换为可用的 rtsp 流源地址（目前只支持264格式的rtsp流，例："rtsp://xxx.xxx.xxx.xxx:xxx/input.264", 其中xxx.xxx.xxx.xxx:xxx为ip和端口号）；

②：将所有“deviceId”字段值替换为实际使用的device的id值，可用的 device id 值可以使用如下命令查看：`npu-smi info`

③：如需配置多路输入视频流，需要配置多个拉流、解码、缩放、推理、序列化插件，然后将多个序列化插件的结果输出发送到串流插件mxpi_parallel2serial（有关串流插件使用请参考《mxVision 用户指南》中“串流插件”章节），最后连接到appsink0插件。

## 4 运行

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
  
  build_zlib.sh脚本详情如下：
  #!/bin/bash
  # Simple log helper functions
  info() { echo -e "\033[1;34m[INFO ][Depend  ] $1\033[1;37m" ; }
  warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
  
  #Build
  fileName="zlib"
  packageFQDN="zlib@1.2.11-h2"
  packageName="zlib"
  cd "$fileName" || {
    warn "cd to ./opensource/$fileName failed"
    exit 254
  }
  
  info "Building dependency $packageFQDN."
  chmod u+x configure
  export LDFLAGS="-Wl,-z,noexecstack,-z,relro,-z,now,-s"
  export CFLAGS="-fPIE -fstack-protector-all -fPIC -Wall -D_GLIBCXX_USE_CXX11_ABI=0"
  export CPPFLAGS="-fPIE -fstack-protector-all -fPIC -Wall -D_GLIBCXX_USE_CXX11_ABI=0"
  export CC=aarch64-linux-gnu-gcc
  ./configure \
    --prefix="$(pwd)/../tmp/$packageName" \
    --shared || {
    warn "Build $packageFQDN failed during autogen"
    exit 254
  }
  
  make -s -j || {
    warn "Build $packageFQDN failed during make"
    exit 254
  }
  
  make install -j || {
    warn "Build $packageFQDN failed during install"
    exit 254
  }
  
  cd ..
  info "Build $packageFQDN done."
  
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

