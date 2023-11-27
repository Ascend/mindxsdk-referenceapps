#  华为AI物理攻击和检测系统

--------------------------

## 1 介绍
    本参考设计主要实现了一个AI物理对抗攻击的检测系统, 通过读取rtsp视频流或本地视频文件, 检测视频帧中是否存在攻击patch, 并将结果发送到本地浏览器web页面展示.

## 1.1 代码目录结构与说明
```
├── camera                             # 视频或视频流处理模块
│   └── camera_process.py              
├── darknet_ob_detector                # 正常目标检测模块
│   ├── detect_ob_by_mxbase.py         
│   ├── utils.py
│   └── model
│       └── coco.names
├── ultrayolo_attack_detector          # 攻击patch检测模块
│   ├── detect_attack_by_mxbase.py     
│   ├── utils.py
│   └── model
│       └── coco.yaml
├── templates
│   └── index.html                     # web页面文件
├── config 
│   └── config.ini                     # 视频流或视频路径配置文件        	            
├── requirements.txt                   # python依赖库
├── args.py                            # 命令解析脚本
├── app.py                             # 主程序入口
├── run.sh                             # 程序启动脚本
└── README.md
```

## 1.2 支持的产品
    Atlas A500 A2

## 1.3 模型获取
| 模型名称       | 下载链接 |
|---------------|------|
| darknet_aarch64_Ascend310B1_input.om | [link](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/mxVision/ai_attacking/darknet_aarch64_Ascend310B1_input.om) |
| ultra_best.om | [link](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/mxVision/ai_attacking/ultra_best.om) |

将下载好的模型darknet_aarch64_Ascend310B1_input.om放在darknet_ob_detector/model目录下, ultra_best.om放在ultrayolo_attack_detector/model目录下

## 2 运行环境及依赖安装

| 名称      | 版本              | 说明                          |
|-----------|------------------|-------------------------------|
| 系统      | Ubuntu 22.04      | 操作系统                      |
| CANN      | 7.0.RC1           | toolkit开发套件包或nnrt推理包  |
| MindX SDK | 5.0.RC3           | mxVison软件包                 |


### 2.1 依赖软件安装
- CANN获取[链接](https://www.hiascend.com/software/cann), 安装[参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/envdeployment/instg/instg_0013.html)
- MindX SDK 获取[链接](https://www.hiascend.com/software/Mindx-sdk), 安装[参考链接](https://www.hiascend.com/document/detail/zh/mind-sdk/50rc3/vision/mxvisionug/mxvisionug_0014.html)

### 2.2 python依赖安装
#### 依赖库及版本
| 名称                   | 版本      |
|------------------------|----------|
| Flask                  | 3.0.0    |
| PyYAML                 | 5.4.1    |
| torch                  | 2.0.1    |
| torchvision            | 0.15.2   |
| opencv-python-headless | 4.7.0.72 |
| nympy                  | 1.25.0   |

### 安装
```shell
pip3 install -r requirements.txt
```

## 3 运行系统

### 3.1 配置输入源
    在config/config.ini文件中修改配置
- 若是视频流则将视频流rtsp地址配置在camera->rtsp, 须用户保证rtsp流存在
- 若是本地视频则将视频路径配置在video->video_path
- 若同时配置了rtsp流地址和本地视频路径, 则只读取rtsp流, 不会读取本地视频.

```
配置示例如下, 仅供参考之用, 不可直接复制使用, 运行程序时需要根据实际情况进行替换:
[camera]
rtsp = rtsp://127.0.0.1:8888/input.264  # 输入为本地rtsp流地址
或者
rtsp = rtsp://usernaem:password@127.0.0.1:8888/h264/ch1/main/av_stream  # 输入为摄像头rtsp流地址

[video]
video_path = ./test.mp4  # 输入为视频文件, 相对路径和绝对路径均可
```

### 3.2 运行app
    运行脚本run.sh
    Options:
      -h/--help            display this help and exit
      -i/--host            listening ip address.
      -p/--port            listening port. The default value is 8888
      -d/--device_id       choose device id. The default value 0.
```shell
chmod u+x run.sh
./run.sh -i 127.0.0.1 -p 8888 -d 0 # 端口号和设备id可根据实际情况自行替换, ip须设置为程序所在的服务器ip地址
```

### 3.3 退出app
    手动在终端执行 ctrl + c 退出程序, 根据实际情况可能需要多次重复执行该操作.

 
## 4 查看结果
    在本地浏览器输入设定的[网址:端口号], 即可查看系统检测的结果

## 5 常见问题

### 5.1 本地浏览器无法访问给定的网址
**问题描述:**
检测程序运行成功后, 访问浏览器, 出现网页故障或无法访问此网站提示.

**解决方案:**
打开本地网络和internet配置, 在代理处将程序运行所在的服务器ip加进去.

### 5.2 执行run.sh报错
**问题描述:**
执行run.sh时, 报错: -bash: ./run.sh: /bin/bash^M bad interpreter*

**解决方案:**
将run.sh文件格式转为unix格式, 下面两种方法都行:
1. 执行 dos2unix run.sh
2. vim打开run.sh, 在编辑模式下输入set ff=unix, 然后wq保存退出.
文件格式修改后, 重新执行run.sh即可.