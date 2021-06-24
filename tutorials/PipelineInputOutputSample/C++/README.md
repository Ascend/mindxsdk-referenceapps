## SDK pipeline 输入输出样例运行

### 介绍
提供的demo，实现 pipeline输入输出调用不同的输入插件的样例运行

### 准备工作
将项目目录从mindxsdk-referenceapps\tutorials\PipelineInputOutputSample\C++移动到运行样例的目录下

### 配置SDK路径
配置CMakeLists.txt 中的'MX_SDK_HOME'环境变量，配置为MindX SDK的安装路径； ${SDK安装路径}替换为用户环境SDK安装路径。
set(MX_SDK_HOME /usr/local/Ascend/mxVision)
/usr/local/Ascend/mxVision 需要替换为自己的SDK安装路径

```
set(MX_SDK_HOME ${SDK安装路径}/mxVision)
```

### 配置环境变量
执行以下命令：

```
export MX_SDK_HOME="${CUR_PATH}/../../.."

export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":"/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64":${LD_LIBRARY_PATH}
```

### 编译运行
在当前路径下新建文件夹 build
进入build 目录
执行 cmake ..
执行 make

返回上级目录 cd ..

执行 ./sample 或者执行 ./sample 参数
其中参数为 0,1,2 参数对应的调用插件可以参考快速指导

### 查看结果
打印 result:hello 则执行成功
