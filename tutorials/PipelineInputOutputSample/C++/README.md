## SDK pipeline 输入输出样例运行

### 介绍
提供的demo，实现 pipeline输入输出调用不同的输入插件的样例运行

### 准备工作
样例获取链接(https://gitee.com/zhangdwe/mindxsdk-referenceapps/tree/master/tutorials/PipelineInputOutputSample/C++)

将项目目录从mindxsdk-referenceapps\tutorials\PipelineInputOutputSample\C++移动到运行样例的目录下


### 配置环境变量
执行以下命令：
```
. ${MX_SDK_HOME}/set_env.sh
# ${MX_SDK_HOME}替换为用户的SDK安装路径
```

### 编译运行
在当前路径下新建文件夹 build
进入build 目录
执行 cmake ..
执行 make

返回上级目录 cd ..

执行 ./IOsample 或者执行 ./IOsample 参数
其中参数为 0,1,2 参数对应的调用插件可以参考快速指导

### 查看结果
打印 result:hello 则执行成功
