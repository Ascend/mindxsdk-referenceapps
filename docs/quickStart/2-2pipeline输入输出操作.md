# 2 pipeline输入输出操作
**本章节中mxManufacture和mxVision可互相替换**
## 2.1 业务流程对接接口介绍
### 2.1.1 SendData-GetResult 数据流图
- 用户使用SendData接口将图片数据发送给appsrc元件，inPluginId指定appsrc的编号，appsrc发送数据给pipline中的其他元件处理，处理结果发送给appsink元件，用户使用GetResult接口获取appsink的数据，其中outPluginId指定appsink编号。
- 本套接口可以用于没有appsrc或appsink元件的场景（即不需要外部输入数据或调接口获取结果），例如将appsrc改成视频取流元件mxpi_rtspsrc时，不需要通过SendData发送数据，输出结果可以用GetResult获取。样例代码请参考SendData-GetResult接口样例代码。

<span style="background-color:#92cddc;"><span style="font-size:18px;">**须知**</span></span>
<ins>当多个线程同时调用SendData接口时，GetResult获取的结果顺序不确定。SendData接口支持多个appsrc输入元件，GetResult支持多个appsink输出元件。</ins>


**图 1-1** SendData-GetResult 数据流图  
![截图.PNG](img/1621941610975.png '1621941610975.png ')


SendData接口输入有三种传入参数方式
以下是Python函数原型与输入参数说明，C++参数类似可参考C++流程管理API参考部分[SendData](https://support.huaweicloud.com/ug-mfac-mindxsdk201/atlasmx_02_0391.html)：
```
def SendData(streamName: bytes, inPluginId: int, dataInput: MxDataInput) -> int:
    pass
```
|参数名  |类型  |说明  |
|--|--|--|
| streamName |bytes  |流的名称。  |
inPluginId |int|目标输入插件Id，即appsrc元件的编号。
dataBuffer|请参考MxDataInput。|待发送的数据。
```
def SendData(streamName: bytes, elementName: bytes, dataInput: MxDataInput) -> int:
    pass
```
|参数名  |类型  |说明  |
|--|--|--|
| streamName |bytes  |流的名称。  |
elementName |bytes|输入插件的名称，只支持appsrc当做输入插件。
dataBuffer|请参考MxDataInput。|待发送的数据。

```
def SendData(streamName: bytes, elementName: bytes, metadataVec: MetadataInputVector, databuffer: MxBufferInput) -> int:
    pass
```
|参数名  |类型  |说明  |
|--|--|--|
| streamName |bytes  |流的名称。  |
elementName |bytes|输入插件的名称，只支持appsrc当做输入插件。
metadataVec|MetadataInputVector|发送的元数据vector。
dataBuffer|MxBufferInput|待发送的buffer数据。




### 2.1.2 SendDataWithUniqueId-GetDataWithUniqueId 数据流图
- 用户使用SendDataWithUniqueId接口将图片数据发送给appsrc元件，inPluginId（“当前固定为0”）指定appsrc元件的编号，返回整数uniqueId给用户，appsrc发送数据给pipline处理，将处理结果以uniqueId为key保存在outputMap中，用户使用GetDataWithUniqueId接口（**使用发送时获得的uniqueId作为入参**）取出与SendDataWithUniqueId对应的推理结果。样例代码请参考SendDataWithUniqueId-GetResultWithUniqueId样例代码。

<span style="background-color:#92cddc;"><span style="font-size:18px;">**须知**</span></span>
<ins>运行该程序之前需要自己手动下载一张包含动物的图片（例如一只狗的图片），并将该图片命名为test.jpg，然后放到python目录下。</ins>

**图 1-2** SendDataWithUniqueId-GetDataWithUniqueId 数据流图  
![截图.PNG](img/1621941652252.png '截图.png')


SendDataWithUniqueId接口输入有两种传入参数方式
以下是Python函数原型与输入参数说明，C++参数类似，可参考C++流程管理API参考部分[SendDataWithUniqueId](https://support.huaweicloud.com/ug-mfac-mindxsdk201/atlasmx_02_0393.html)：
```
def SendDataWithUniqueId(streamName: bytes, inPluginId: int, dataInput: MxDataInput) -> int:
    pass
```
|参数名  |类型  |说明  |
|--|--|--|
| streamName |bytes  |流的名称。  |
inPluginId |int|目标输入插件Id，即appsrc元件的编号。
dataBuffer|请参考MxDataInput。|待发送的数据。
```
def SendDataWithUniqueId(streamName: bytes, elementName: bytes, dataInput: MxDataInput) -> int:
    pass
```
|参数名  |类型  |说明  |
|--|--|--|
| streamName |bytes  |流的名称。  |
elementName | bytes |输入插件的名称，只支持appsrc当做输入插件。
dataBuffer|请参考MxDataInput。|待发送的数据。



### 2.1.3 SendProtobuf-GetProtobuf 数据流图
用户使用接口SendProtobuf将protobuf数据和key（用于将protobuf数据挂载至元数据中，使用该数据的元件可以通过这个key获取数据）批量或单个发送给appsrc元件，inPluginId指定appsrc的编号，appsrc发送数据给其他元件处理。元件处理完数据后，以元件名为key将处理结果保存至元数据中，最后通过GetProtobuf接口从元数据中取出想要获取的元件结果，输入一组key，便能获取key对应的protobuf数据。
本套接口可以用于没有appsrc或appsink元件的场景（即不需要外部输入数据或调接口获取结果），例如将appsrc改成视频取流元件mxpi_rtspsrc时，不需要通过SendProtobuf发送数据，输出结果可以用GetProtobuf获取。样例代码请参考SendProtobuf-GetProtobuf样例代码。

<span style="background-color:#92cddc;"><span style="font-size:18px;">**须知**</span></span>
<ins>当多个线程同时调用SendProtobuf接口时，GetProtobuf获取的结果顺序不确定。SendProtobuf接口支持多个appsrc输入插件，GetProtobuf支持多个appsink输出元件。</ins>

**图 1-3** SendProtobuf-GetProtobuf 数据流图  
![截图.PNG](img/1622173570842.png '截图.png')

SendProtobuf接口输入有两种传入参数方式
以下是Python函数原型与输入参数说明，C++参数类似，可参考C++流程管理API参考部分[SendProtobuf](https://support.huaweicloud.com/ug-mfac-mindxsdk201/atlasmx_02_0395.html)：
```
def SendProtobuf(streamName: bytes, inPluginId: int, protobufVec: list) -> int:
    pass
```
|参数名  |类型  |说明  |
|--|--|--|
| streamName |bytes  |流的名称。  |
inPluginId |int|目标输入插件Id，即appsrc元件的编号。
protobufVec| MxProtobufIn list |发送的MxProtobufIn列表，将protobuf的key、type、value发给mxstream，其中value是将protobuf序列化后的bytes。
```
def SendProtobuf(streamName: bytes, elementName: bytes, protobufVec: list) -> int:
    pass
```
|参数名  |类型  |说明  |
|--|--|--|
| streamName |bytes  |流的名称。  |
elementName | bytes |输入插件的名称，只支持appsrc当做输入插件。
protobufVec| MxProtobufIn list |发送的MxProtobufIn列表，将protobuf的key、type、value发给mxstream，其中value是将protobuf序列化后的bytes。




**表 1-2** 接口对比表

|接口名称  | 输入数据类型  | 输出数据类型 | 输入和输出是否有序 | 是否支持多输入多输出 | 是否必须配对使用| 使用场景|
|--|--|--|--|--|--|--|
SendData-GetResult |图片|序列化输出结果|否|是|否| 支持单线程有序或多线程输出结果乱序。|
SendDataWithUniqueId-GetDataWithUniqueId|图片|序列化输出结果|是 |否 |是 |支持输入输出有序的单线程或多线程。例如，创建推理服务，不同客户端并发向其发送请求。|
SendProtobuf-GetProtobuf |protobuf|protobuf|否| 是 |否 |支持单线程有序或多线程输出结果乱序。


## 2.2 运行接口样例
### 2.2.1 Clion运行C++样例代码
以下步骤是Clion运行pipeline输入输出样例的步骤说明，设置Clion操作相关的问题可以参考开发环境章节。样例运行也可参考README.md。
#### 步骤 1 Clion开发环境搭建
参见 IDE开发环境搭建--基于Clion开发调试章节，实现安装clion和远程环境连接。

C++ 样例在（[项目目录地址](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/PipelineInputOutputSample/C++)）路径下。
将路径下的整个C++文件夹下载到本地用Clion打开该项目文件。

#### 步骤2 配置环境变量
可参考2-1图像检测样例运行章节

clion任务栏 Run->Edit Configurations->Environment variables 添加环境变量

将${SDK安装路径}替换为自己的SDK安装路径；将${MX_SDK_HOME}替换成路径
```
MX_SDK_HOME=${SDK安装路径}/mxManufacture
LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
```
 Example
```
MX_SDK_HOME=/home/*****/home/work/MindX_SDK/mxManufacture
LD_LIBRARY_PATH=/home/*****/home/work/MindX_SDK/mxManufacture/lib:/home/*****/home/work/MindX_SDK/mxManufacture/opensource/lib:/home/*****/home/work/MindX_SDK/mxManufacture/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
GST_PLUGIN_SCANNER=/home/*****/home/work/MindX_SDK/mxManufacture/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
GST_PLUGIN_PATH=/home/*****/home/work/MindX_SDK/mxManufacture/opensource/lib/gstreamer-1.0:/home/*****/home/work/MindX_SDK/mxManufacture/lib/plugins

```

#### 步骤 3 配置SDK路径

配置CMakeLists.txt中的MX_SDK_HOME环境变量，配置安装了MindX SDK的路径

set(MX_SDK_HOME ${SDK安装路径}/mxManufacture)  

![image.png](img/1623388882981.png 'image.png')  


配置LD_LIBRARY_PATH环境变量
clion任务栏 file->settings->Build Execution Deployment -> Cmake -> Environment 选择文件图标，加号添加环境变量
变量名 LD_LIBRARY_PATH，值/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/  
![image.png](img/1623749681297.png 'image.png')
#### 步骤 4 编译 运行

执行Clion->Build->Build '工程名'
执行Clion->Run->Run"xxxx"


####  步骤 5 切换输入插件
clion任务栏 Run->Edit Configurations->Environment Program arguments

Program arguments 传入[0,1,2]可以切换调用接口，对应关系可以见表 1-1 中的对应关系。  
![image.png](img/1623236074913.png 'image.png')  

**表 1-1** 对应关系表

|Program arguments（常量）  |调用接口 | 
|--|--|
| 0 | SendData 接口|(streamName: bytes, inPluginId: int, dataInput: MxDataInput)|  
| 1 | SendDataWithUniqueId 接口|(streamName: bytes, elementName: bytes, dataInput: MxDataInput)|  
| 2 | SendProtobuffer 接口|(streamName: bytes, elementName: bytes, metadataVec: MetadataInputVector, databuffer: MxBufferInput)|  
 
####  步骤 6 查看输出结果
结果打印 result:hello 则运行成功  
![image.png](img/1623236387023.png 'image.png')


#### 补充：处理头文件报错
如果头文件报错可以将配置好CMakeList.txt 右键执行**Reload CMake Project**操作  
![image.png](img/1623224745826.png 'image.png')




### 2.2.2 运行Python样例代码
以下步骤是pyCharm运行pipeline输入输出样例的步骤说明，设置pyCharm操作相关的问题可以参考开发环境章节。样例的命令行运行可参考README.md。
#### 步骤 1 pyCharm开发环境搭建
参见 IDE开发环境搭建--基于pyCharm开发调试章节，实现安装pyCharm和远程环境连接。

python 样例在（[项目目录地址](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/PipelineInputOutputSample/python)）路径下。
将路径下的整个python文件夹下载到本地用pyCharm打开该项目文件。

#### 步骤 2 配置SDK路径

配置pyCharm中MX_SDK_HOME和LD_LIBRARY_PATH环境变量，参考章节1 IDE开发环境搭建

#### 步骤 3 编译 运行

执行pyCharm->Run->Run"main"

####  步骤 4 切换输入插件
样例默认使用的是SendData的第一种传参方式即SendData(streamName: bytes, inPluginId: int, dataInput: MxDataInput)，如果需要切换其他输入方式需要执行以下操作。命令行切换可以参考README.md

- pyCharm 任务栏 Run->Edit Configurations->Parameters

Program arguments 传入[1-9]可以切换调用接口，对应关系可以见表 1-2 中的对应关系。  
![image.png](img/1623236074913.png 'image.png')

**表 1-2** 对应关系表

|Program arguments（常量）  |调用接口 | 传入参数|
|--|--|--|
|1 | SendData 接口|(streamName: bytes, inPluginId: int, dataInput: MxDataInput)|  
| 2 |SendData 接口 |(streamName: bytes, elementName: bytes, dataInput: MxDataInput)|  
| 3 |SendData 接口 |(streamName: bytes, elementName: bytes, metadataVec: MetadataInputVector, databuffer: MxBufferInput)|  
| 4 | SendDataWithUniqueId 接口|(streamName: bytes, inPluginId: int, dataInput: MxDataInput)|  
| 5 |SendDataWithUniqueId 接口|(streamName: bytes, elementName: bytes, dataInput: MxDataInput)|  
|6 | SendProtobuffer 接口|(streamName: bytes, inPluginId: int, protobufVec: list)|  
| 7 | SendProtobuffer 接口|(streamName: bytes, elementName: bytes, protobufVec: list)|  



 
####  步骤 5 查看输出结果
结果打印 result:success 则运行成功  

![image.png](img/1623324184025.png 'image.png')
