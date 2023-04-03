## SDK pipeline 输入输出样例运行

### 介绍
提供的demo，实现输入输出插件的样例运行

### 准备工作
样例获取链接(https://gitee.com/zhangdwe/mindxsdk-referenceapps/tree/master/tutorials/PipelineInputOutputSample/python)

将样例目录python 从mxsdkreferenceapps/tutorials/pipelineInputOuputSample文件夹下 移动到${SDK安装路径}/mxVision/samples/mxVision/python/路径下

可以使用mv 命令
进入到移动后的工程路径下

### 配置环境变量

执行以下命令：
```
. ${MX_SDK_HOME}/set_env.sh
# ${MX_SDK_HOME}替换为用户的SDK安装路径
```


### 运行

python3 main.py 参数
参数可以传入1-7 的数字 参数对应关系可以参考快速指导文档

```
python3 main.py 1
```

### 查看结果
打印 result:Success