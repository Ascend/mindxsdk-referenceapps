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
