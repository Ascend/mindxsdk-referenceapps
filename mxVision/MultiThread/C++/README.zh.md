# Multi Thread

## 1.介绍

- 多线程调用SDK的发送和接收数据接口函数

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
| mxVision | 0.2    |

## 3.配置

- 当前测试用例包含2个样例:
1号样例调用EasyStream.pipeline文件
2号样例调用EasyStream_protobuf.pipeline文件

### 1.1 修改业务及device ID

以样例1为例：

- 可以通过修改EasyStream.pipeline文件中插件的个数和属性，修改业务功能（业务名称为“detection0，detection1”，以此类推，方便调用），设定"deviceId":"0"，绑定该业务使用的芯片号（不同的业务可以使用相同的芯片）
- 在main.cpp的TestMultiThread函数中，修改threadCount变量的值，实现多线程功能,修改streamName[i]的值，则是调用不同芯片对应的业务

注意：
- 调用样例2时，需要修改TestSendProbuf函数中width[i]和height[i]的值与业务中的模型输入的宽高一致
- 每次main.cpp文件被修改后，都需要重新执行`./build.sh`

### 3.2 输入数据

- 将测试图片拷贝到`MultiThread`下的`picture`目录

注意：
测试用例当前仅支持分辨率在（32,8192）之间的`.jpg`格式的文件,其他格式文件自动过滤

### 3.3 预准备

脚本转换为unix格式
```bash
sed -i 's/\r$//' ./*.sh
```
给script目录下的脚本添加执行权限

### 3.4 模型转换
yolov3模型下载参考华为昇腾社区[ModelZoo](https://www.hiascend.com/zh/software/modelzoo/detail/1/ba2a4c054a094ef595da288ecbc7d7b4)  
使用以下命令进行转换，请注意aipp配置文件名，此处使用的为自带sample中的相关文件（{Mind_SDK安装路径}/mxVision/samples/mxVision/models/yolov3/）

```
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0"

# 说明：若用例执行在310B上，则--soc_version=Ascend310需修改为Ascend310B1
```

## 4.编译

- 配置环境变量
```bash
export MX_SDK_HOME=${安装路径}/mxVision
```
- 执行`./build.sh`，在dist文件夹中会生成`mxVisionMultiThread`。
```bash
./build.sh
```

## 5.运行

### 5.1 运行前配置

- 在`MultiThread`下创建`models`目录
```bash
mkdir models
```
- 获取`EasyStream.pipeline`中推理所需的om等文件，并拷贝到`models`路径下

### 5.2 运行

- 进入C++目录，执行`bash run.sh 0`,运行程序
```bash
bash run.sh 0
```
注意：
0 表示执行main.cpp中的1号样例，还可以输入`bash run.sh 1`执行2号样例（其他参数表示默认2号样例）

### 5.3 结果
- 样例1运行结果格式如下：

`GetResult: {"MxpiObject":[{"classVec":[{"classId":15,"className":"cat","confidence":0.98471146799999998,"headerVec":[]}],"x0":86.182579000000004,"x1":247.33078,"y0":86.406199999999998,"y1":442.07312000000002},{"classVec":[{"classId":16,"className":"dog","confidence":0.99579948200000001,"headerVec":[]}],"x0":220.453766,"x1":434.736786,"y0":132.42176799999999,"y1":466.86648600000001}]}
`

- 样例2运行结果格式如下：

`value(detection2) = objectVec {
   headerVec {
     dataSource: "mxpi_datatransfer0"
   }
   x0: 57.241951
   y0: 67.1379089
   x1: 189.799133
   y1: 374.405853
   classVec {
     classId: 15
     className: "cat"
     confidence: 0.550405
   }
 }
 objectVec {
   headerVec {
     dataSource: "mxpi_datatransfer0"
   }
   x0: 180.270584
   y0: 74.9994659
   x1: 351.540314
   y1: 401.286896
   classVec {
     classId: 16
     className: "dog"
     confidence: 0.994807
   }
 }
 }
`
