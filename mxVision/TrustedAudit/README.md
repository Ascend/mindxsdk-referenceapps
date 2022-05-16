# 可信审计

## 1 简介

随着各种AI应用加速“嵌入”到各行各业，AI算力已经成为现代文明的重要基础设施，并成为“新基建”的重要方向。但任何有望变革人类社会的新技术都必然会带来广泛的关注，只有确保AI是安全的、可信的，人们才能信任它的决策建议，AI才能得到更广泛的应用。
2021年，随着欧盟《人工智能法规（AI Regulation）》草案的发布，揭开了世界各国对AI可信监管的序幕。在《人工智能法案》中，对高风险AI系统明确提出了数据治理，责任追溯，准确性、健壮性和网络完全方面的要求。其中，在AI整个生命周期中实现日志记录与责任追溯，成为推动可信AI相关各方明确责任的重要抓手。但AI在训练、部署、运行过程中产生的日志数量巨大，并且往往涉及到不同的利益相关方，如何保障海量的训练、推理日志不被篡改而丧失责任追溯的能力，成果为了一个重要的技术挑战。
为解决该问题，本开发样例基于Mindx SDK实现了面向AI监管要求的可信日志（完整性保护）。核心思想是允许将海量的日志原文存储于普通数据库，但在接收日志的过程中同步生成日志的完整性证据，而将日志的密码学（完整性）证据存储到高安数据库。具体而言，是基于用户行为产生的日志，采用区块链中的Merkle tree机制，生成其对应的密码学完整性证据，将日志原文和Merkle存储于低安全区数据库（如elasticsearch数据库）；而将作为完整性证据的Merkle树树根存储于高安全区数据库（如Gauss数据库）。平台管理员可通过比较低安全区的原始日志、验证路径，和高安全区的Merkle树树根，判断原始日志是否被篡改。本样例以昇腾Atlas310卡为主要的硬件平台，主要支持以下功能：
1）MindX日志监控与处理：监控MindX日志文件，发送给可信审计服务端
2）其他日志源（如网关）处理：支持其他日志源日志发送给可信审计服务端
3）日志可信存储与审计：可信审计服务端为日志构造Merkle树，分别存储原始日志、对应的完整性证据至低、高安全区数据库


## 2 环境依赖

- 支持的硬件形态和操作系统版本

| 硬件形态          | 操作系统版本     |
| -----------------| --------------- |
| Atlas 310I 推理卡 | Ubuntu 18.04.1 |

- 软件依赖

| 软件名称             | 版本     |
| -------------------- | -------- |
| MindX SDK            | 2.0.2    |
| docker               | 20.10.12 |
| Python               | 3.7.5    |
| python扩展包psutil   | 5.8.0    |
| python扩展包watchdog | 2.1.6    |



## 3 代码主要目录介绍

本代码仓名称为mxSdkReferenceApps，工程目录如下图所示：

```
├── mxVision
│   ├── TrustedAudit
│   |   ├── docker
│   |   │   ├── docker_run.sh
│   |   │   ├── Dockerfile_es
│   |   │   ├── Dockerfile_opengauss
│   |   │   └── Dockerfile_python
│   |   ├── plugins
│   |   │   ├── build.sh
│   |   │   ├── CMakeLists.txt
│   |   │   ├── MxpiTrustedAuditPlugin.cpp
│   |   │   └── MxpiTrustedAuditPlugin.h
│   |   ├── trusted_audit
│   |   │   ├── mindx
│   |   │   │   ├── kill_watcher.py
│   |   │   │   └── mindx_watcher_and_sender.py
│   |   │   ├── src
│   |   │   │   ├── database_init.py
│   |   │   │   ├── es_database_operate.py
│   |   │   │   ├── full_audit.py
│   |   │   │   ├── full_search.py
│   |   │   │   ├── gauss_database_operate.py
│   |   │   │   ├── merkle.py
│   |   │   │   ├── server_config.py
│   |   │   │   ├── tranlog_audit_serv.py
│   |   │   │   ├── user_audit.py
│   |   │   │   └── user_search.py
│   |   │   ├── main_trusted_audit.py
│   |   │   ├── TrustedAudit.pipeline
│   |   │   ├── test_a.py
│   |   │   ├── test_b.py
│   |   │   ├── test_c.py
│   |   │   └── test_d.py
│   |   └── README.md
```


## 4 配置

**步骤1：** 参考安装教程《mxVision 用户指南》安装 mxVision SDK。

**步骤2：** 配置 mxVision SDK 环境变量。

```
vi ~/.bashrc
export MX_SDK_HOME=${安装路径}/mxVision
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH="${MX_SDK_HOME}/python:${PYTHONPATH}" 
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PULGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins"
source ~/.bashrc
```

注：本例中mxVision SDK安装路径为 /work/MindX_SDK。环境变量介绍：
- MX_SDK_HOME为SDK安装路径
- LD_LIBRARY_PATH为lib库路径
- PYTHONPATH为python环境路径

***步骤3***:确保mindx的日志层级为Info
修改${MX_SDK_HOME}/config/logging.conf文件的global_level变量为0


## 5 准备
### 5.1 Docker容器启动

***步骤1***:进入docker文件所在目录并编译

```
cd 项目所在目录/docker
bash docker_run.sh
```
(若首次运行，请确保系统正确联网，以便从互联网下载标准ElasticSearch/python/opengauss docker镜像，如需要代理，请根据实际情况修改Dockerfile中的代理配置；编译脚本会自动建立gauss数据库、elasticsearch数据库、python运行环境三个容器，并配置网络环境；最后初始化数据库，建立对应的库、表）

***步骤2***:查看数据库初始化情况

```
cat /tmp/database_init.log
```
(初始化数据库，建立对应的库、表的过程存储在/tmp/database_init.log文件中）

### 5.2 插件编译
进入插件文件所在目录并编译

```
cd 项目所在目录/plugins
bash build.sh
```
（编译脚本会自动复制编译好的libmxpi_trustedauditplugin.so文件到${MX_SDK_HOME}/lib/plugins/路径）

### 5.3 插件配置
用户可根据Mindx安装路径更改Mindx源日志输出路径，配置文件为：项目所在目录/trusted_audit/test/TrustedAudit.pipeline
```
"mxpi_trustedauditplugin":{
    "props":{
        "descriptionMessage": "Trusted Audit Plugin Message",
        "originalLogsPath": "/work/mindx_sdk/mxVision/logs"
    }
}
```
originalLogsPath字段表示mindx源日志文件所在位置，默认为/work/mindx_sdk/mxVision/logs

同时修改，项目所在目录/trusted_audit/mindx/mindx_watcher_and_sender.py，第152行变量folder_name，默认为"/work/mindx_sdk/mxVision/logs"

## 6 运行
### 运行可信日志服务端主程序插件

***步骤1***:运行
```
cd 项目所在目录/trusted_audit/
python main_trusted_audit.py
```
（该程序会启动gauss数据库、elastic数据库、python环境所在的三个docker环境；启动python环境中的可信日志主服务；启动watcher监控mindx源日志文件夹；该文件的命令行输出类似下述内容后**暂停**）

```
MxpiTrustedAuditPlugin::Init start.
docker exec container_python_1.0 python -u /home/tranlog_audit_serv.py >> /tmp/server.log 2>&1 &
python -u 项目所在目录/trusted_audit/mindx/mindx_watcher_and_sender.py >> /tmp/watcher.log 2>&1 &
Creates streams successfully.
```
（注意，此时主函数暂停，该shell窗口等待最后ctrl+c关闭，后续操作在其他shell窗口中完成）

***步骤2***:在另一个shell窗口中检查主函数运行情况
```
cat /tmp/server.log
```
主函数运行情况如下
```
可信日志服务端, docker制作日期...
...
收到落库请求，包含73行mindx日志，写入文件队列，时间：...
```

***步骤3***:在另一个shell窗口中检查watcher监控器运行情况
```
cat /tmp/watcher.log
```
watcher监控器运行情况如下
```
watcher is running...datetime is: ...
info: file_name mxsdk.log.info.... 本次读取 73 行
... 时检测到文件 mxsdk.log.info... 变化，当前已读取到 74 行
发送结果 <Response [200]> upload ... items success when ...
```

## 7 可信日志审计测试
### 7.1 测试发送25条网关日志到主程序
***步骤1***:在另一个shell窗口运行
```
cd 项目所在目录/trusted_audit/
python test_a.py
```
该shell窗口显示

```
mock日志含1个用户，共计 25 条日志
模仿网关生成mock的用户名和起止时间为 ae6791e7330ded894b2e60fb2e9ab444 2021-12-17 15:09:17.364057 2021-12-17 15:09:41.364057
```


***步骤2***:检查主函数运行情况
```
cat /tmp/server.log
```
主函数运行情况如下
```
收到落库请求，包含 25 行网关日志，写入文件队列，时间：...
定时器已关闭，开启 10 秒定时器
定时器 10 秒等待时间够了，准备处理
准备处理第...个chunk，含25条日志，当前时间...
处理完第...个chunk，含25条日志，当前时间...
```
### 7.2 按用户查询之前发送的网关日志

运行按用户查询的用例程序，注意输入正确的用户id和起止时间；查询窗口为每页10条，请求第1页
```
cd 项目所在目录/trusted_audit/
python test_b.py ae6791e7330ded894b2e60fb2e9ab444 2021-12-17 15:09:17.364057 2021-12-17 15:09:41.364057 1 10
```
该shell窗口显示

```
检索到 25 条日志，第 1 页的 10 条日志对应的log_id为 ... 验证结果为 ...
```
（注意：按用户查找时，标识符0表示该条目日志正确；标识符1表示该条目被篡改；后续用例将展示被篡改的效果）


### 7.3 按时间段查询之前发送的网关日志

运行按时间段查询的用例程序，注意输入正确的起止时间；查询窗口为每页10条，请求第1页
```
cd 项目所在目录/trusted_audit/
python test_c.py 2021-12-17 15:09:17.364057 2021-12-17 15:09:41.364057 1 10
```
该shell窗口显示

```
检索到 25 条日志，第 1 页的 10 条日志对应的log_id为 ... 验证结果为 ...
```
（注意：按时间段查找时，标识符0表示该条目日志正确；标识符2表示该Merkle树对应条目日志中某些条目被篡改，需要进一步按用户查找定位错误日志；后续用例将展示被篡改的效果）

### 7.4 手动篡改日志后检查效果

***步骤1***:手动更改日志，执行
```
cd 项目所在目录/trusted_audit/
python test_d.py ae6791e7330ded894b2e60fb2e9ab444 hello_world
```
该步骤将更改低安全区数据库中用户ae6791e7330ded894b2e60fb2e9ab444对应日志的第5条内容为hello_world，shell信息提示如下

```
log_id为...的ES数据中item_raw_content字段已被修改为{'key': 'hello_world'}
```

***步骤2***:运行按用户查询的用例程序，注意**起止时间要与7.2的用例些许不同**
```
cd 项目所在目录/trusted_audit/test
python test_b.py ae6791e7330ded894b2e60fb2e9ab444 2021-12-17 15:09:16.364057 2021-12-17 15:09:42.364057 1 10
```
该shell窗口显示

```
检索到 25 条日志，第 1 页的 10 条日志对应的log_id为 ... 验证结果为 ...
```
（注意：按用户查找时，标识符0表示该条目日志正确；标识符1表示该条目被篡改）

***步骤3***:运行按时间段查询的用例程序，注意**起止时间要与7.2的用例些许不同**
```
cd 项目所在目录/trusted_audit/test
python test_c.py 2021-12-17 15:09:16.364057 2021-12-17 15:09:42.364057 1 10
```
该shell窗口显示

```
检索到 25 条日志，第 1 页的 10 条日志对应的log_id为 ... 验证结果为 ...
```
（注意：按时间段查找时，标识符0表示该条目日志正确；标识符2表示该Merkle树对应条目日志中某些条目被篡改，需要进一步按用户查找定位错误日志）

## 8 停止插件
回到之前启动插件的shell插件，按下ctrl+c键停止该插件运行，shell窗口输出
```
Destroys the stream successfully.
```
主程序停止插件时会自动关闭watcher监控mindx源日志的程序，并清空mindx源日志文件夹；可如下检查监控器关闭情况；执行
```
cat /tmp/kill_watcher.log
```
观察到shell输出
```
... watcher已经启动，停止watcher进程，退出
```