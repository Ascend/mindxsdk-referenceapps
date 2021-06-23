# IDE开发环境搭建--Clion

以图像检测项目sample为例演示基于本地ID--Clion的开发调试。

## 1.2.1 获得项目文件
**情况1** 当项目文件在远程环境中时，需要从远程环境中将想要开发的C++项目文件下载到本地，用CLion打开项目，并按以下操作同步本地与远程环境。  
![1.png](img/1623219436881.png '1.png')

**情况2** 若项目文件就在本地环境中，则用Clion打开项目文件，并按照以下步骤连接本地与远程环境，将项目文件上传到远程环境中去。


## 1.2.2 连接远程环境

**步骤1**   配置远程服务器toolchains。选择File->Settings，在Build Execution Deployment -> toolchains创建Remote Host会话  
![1_1.png](img/1622100078616.png '1_1.png')

**步骤2**   配置远程服务器SSH连接  
![3.png](img/1622528329436.png '3.png')

**步骤3**   配置远程服务器cmake、make、gcc、g++路径。具体路径可以通过在远程服务器上输入*which [cmake\make\gcc\g++]* 获取对应的路径  
![1_3·.png](img/1622101236396.png '1_3·.png')

**步骤4**  点开Build Execution Deployment -> Deployment 修改上一步创建的Remote-Host中的远程映射路径： 
![2.png](img/1623220168055.png '2.png')  
首次创建好连接路径后CLion会自动同步两端文件。
**此后若修改文件，可通过右击工程 Deployment->Upload to...，把项目上传到设置的远程路径中去；通过右击工程 Deployment->Download from...从远程环境中将修改后的项目下载到本地路径中。**  
![3.png](img/1623220230223.png '3.png')  
同步后的两端文件内容保持一致：  
![4.png](img/1623220694823.png '4.png')

## 1.2.3 远程运行调试文件
**步骤1**  加载CMakeLists.txt文件
参考[CMake学习](wiki/常见资料获取/CMake学习)修改CMakeLists.txt文件。
修改好CMakeLists文件后，重新加载CMakeLists文件,cmake-build-debug构建目录也会被重新生成。  
  ![1.png](img/1622518642593.png '1.png')
  
  若是自己新建的项目，在编写好CMakeLists文件后，右键点击文件，点击Load CMake Project加载CMake：  
![12.png](img/1622259348404.png '12.png')


cmake-build-debug构建目录会在首次加载完成CMakeLists文件后出现。

在Settings -> Build, Execution, Deployment -> CMake -> Environment中添加环境变量：
```
LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/ 
```
然后便可以点击Build编译该项目了。出现如下语句则表示编译成功。  
![5.png](img/1623220827671.png '5.png')

 **步骤2**  项目运行
打开Run -> Run Configurations配置运行参数。配置执行路径和执行时的环境变量。
主要有以下的环境变量：
**注意${}中的路径需要替换为实际的值，否则Clion无法读取对应目录**
```
MX_SDK_HOME=${SDK安装路径}
LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
```
操作如下图所示：  
![6.png](img/1623221233288.png '6.png')

### 补充
可在远程环境中使用*env*指令查看所需环境变量值。
请仔细检查自己的环境变量与上文所给的环境变量是否有差别，若缺少路径，执行*vi .bashrc*命令，在.bashrc文件中添加缺少的环境变量。
保存退出后，执行*source ~/.bashrc*命令使环境变量生效。

配置好环境变量，点击Run运行程序（项目运行所需操作请参考[图像检测sample样例运行](wiki/2初级开发/2-1图像检测sample样例运行)）。


**步骤3**  断点调试

对测试代码加断点，在左端行号前面点击，会出现红色断点  
![7.png](img/1623221481373.png '7.png')

点击任务栏的RUN->Debug(shift + F9)程序会执行到上一步断点位置，可以观察当前变量的值，后续可以使用单步调试。  
![8.png](img/1623221646773.png '8.png')

若断点不生效，在CMakeLists文件中加入以下语句，设置编译类型为Dedug。
```
set(CMAKE_BUILD_TYPE Debug)
```


# IDE开发环境搭建--pyCharm

## 获取项目
当项目文件在远程环境中时，需要从远程环境中将想要开发的Python项目文件下载到本地，用pyCharm打开项目，并按以下操作同步本地与远程环境。
## 连接远程环境

**步骤1**   配置远程服务器的Python解释器。选择File->Settings，在Project ${项目名} -> Python Interpreter 选择设置 -> add
![image.png](img/1623309211218.png 'image.png')

选择SSH Interpreter 在Host 位置填写远程服务器IP地址，Username填写用户名   点击Next   等连接成功输入远程登录密码。

![image.png](img/1623309361995.png 'image.png')


**步骤2**  配置本地目录和远程映射路径。选择File->Settings，在Project ${项目名} -> Path mappings 点击文件符号。  
![image.png](img/1623316129521.png 'image.png')  
进入Edit Project Path Mappings 点击加号  添加本地项目路径和远程映射路径  
![image.png](img/1623316215637.png 'image.png')



**步骤3**  配置环境变量 **LD_LIBRARY_PATH** 和 **MX_SDK_HOME** 。 打开Run -> Run Configurations配置运行参数。
选择 Environment variables 后的图标
![image.png](img/1623316788931.png 'image.png')

点击加号 新增LD_LIBRARY_PATH 和MX_SDK_HOME 名字的环境变量。
其中MX_SDK_HOME环境变量设置为远程SDK安装路径，可在远程环境用pwd 命令查看路径
- eg:  /home/目录/home/work/MindX_SDK/mxManufacture

LD_LIBRARY_PATH 环境变量路径设置为以下路径， 需要将${MX_SDK_HOME}替换为实际路径。
```
${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:。
```

![image.png](img/1623316885642.png 'image.png')

## 补充：pyCharm中添加路径解决导入到报红的问题

- 选择File->Settings，在Project ${项目名} -> Python Interpreter 选择设置 -> Show All
![image.png](img/1623315719375.png 'image.png')  


- 选择远程Remote 上的 的符号 添加SDK安装目录下的Python路径  
![image.png](img/1623315806818.png 'image.png')  
![image.png](img/1623755172684.png 'image.png')  


