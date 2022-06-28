# Rtsp Server 

## 1.介绍

- 本样例为加密传输服务端起视频流。

## 2.环境依赖

- 支持的硬件形态和操作系统版本

| 硬件形态                             | 操作系统版本   |
| ------------------------------------ | -------------- |
| x86_64+Atlas 300I 推理卡（型号3010） | Ubuntu 18.04.1 |
| x86_64+Atlas 300I 推理卡（型号3010） | CentOS 7.6     |
| ARM+Atlas 300I 推理卡 （型号3000）   | Ubuntu 18.04.1 |
| ARM+Atlas 300I 推理卡 （型号3000）   | CentOS 7.6     |

- 软件依赖

| 软件名称 | 版本   |
| -------- | ------ |
| cmake    | 3.5.1+ |
| mxVision | 2.0.3+ |

## 3.编译

### 3.1 依赖库安装

    -选择如下依赖库:
    libgstreamer1.0-dev libgstrtspserver-1.0-dev 
    libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools

    例如，ubuntu按如下方式安装:
    apt install libgstreamer1.0-dev libgstrtspserver-1.0-dev
    apt install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools

### 3.2 源码下载

    git clone https://github.com/enthusiasticgeek/gstreamer-rtsp-ssl-example

### 3.3 编译

    cd xxx/gstreamer-rtsp-ssl-example
    make
    生成rtsp_server可执行程序，则说明编译成功。

## 4 运行

### 4.1  配置文件修改

    运行前需配置证书、用户名、密码、起流端口及节点等信息,具体如下所示:
    cd xxx/gstreamer-rtsp-ssl-example
    vi rtsp_parameters.conf

    ```
    RTSP_CA_CERT_PEM=xxx/ca.crt
    RTSP_CERT_PEM=xxx/server.crt
    RTSP_CERT_KEY=xxx/server_no.crt
    RTSP_SERVER_PORT=8554
    RTSP_SERVER_MOUNT_POINT=test
    RTSP_USERNAME=user
    RTSP_PASSWORD=password
    ```

    注:起流程序默认不带解密功能，需使用openssl工具手动解密，命令如下:
    openssl rsa -in server.key -out server_no.crt
    生成server_no.crt，则说明成功。

    证书ca.crt、server.crt的制作，请参考mxVision手册证书制作章节。

### 4.2 运行

    cd xxx/gstreamer-rtsp-ssl-example
    ./rtsp_server
    起流成功后提示: stream ready at rtsps://127.0.0.1:8554/test
