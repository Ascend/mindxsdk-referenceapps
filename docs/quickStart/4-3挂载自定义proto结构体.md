# 挂载自定义proto结构体开发指导

本章节将指导在SDK自带的图像分类识别样例中挂载自定义proto结构并输出的流程。  
在执行本样例前，应已成功远程部署并运行4-1章节的自定义插件样例。  
[点击跳转代码样例](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/sampleCustomProto)

## 自定义proto结构说明

stream支持挂载特定格式的proto结构体到metadata中并向下传递  
必须具备以下结构：  
- 使用list封装
- 包含可重复MxpiMetaHeader头

以下为最简化示例，该文件与样例代码中proto相同。
```protobuffer
syntax = "proto3";
package sampleproto;
message MxpiMetaHeader
{
    string parentName = 1;
    int32 memberId = 2;
    string dataSource = 3;
}

message MxpiSampleProto
{
    repeated MxpiMetaHeader headerVec = 1;
    int32 intSample = 2;
    string stringSample = 3;
    bytes bytesSample = 4;
}
message MxpiSampleProtoList
{
    repeated MxpiSampleProto sampleProtoVec = 1;
}
```

## 项目准备  
**步骤1** 准备基础工程  
参考[4-1插件开发调试指导](4-1插件开发调试指导.md)部署自定义插件样例，示例使用sampleCustomProto作为工程名，远程目录名同样为sampleCustomProto。  

部署proto文件夹至工程中，完成后目录结构和项目名称如下：  
![image.png](img/202107201810.png 'image.png')  
![image.png](img/202107201812.png 'image.png')  
- 更改mxVision/C++/main.cpp中94行所使用的pipeline为样例中的Sample_proto.pipeline
- 更改mxVision/python/main.py中32行使用的pipeline为样例中的Sample_proto.pipeline
- 相比4-1样例中的SamplePlugin.pipeline，原24行的自定义的插件现在位于42行，且修改了51行`"outputDataKeys": "mxpi_sampleproto"`，使输出的结果为自定义的proto结构
>该文件位于样例根目录，但代码中实际指向mxVision/pipeline文件夹下，这是为了与使用原有Sample.pipeline的样例统一目录。实际使用时复制pipeline文件或更改代码中的路径均可  

**步骤2** 添加proto相关文件编译  
修改顶层目录的cmakelist.txt，参考如下：
```cmake
cmake_minimum_required(VERSION 3.6)
project(mxpi_samplecustomproto)

#该语句中%MX_SDK_HOME%根据实际SDK安装位置修改，可通过在终端运行env命令查看
set(ENV{MX_SDK_HOME} %MX_SDK_HOME%)

add_subdirectory("./mxVision/C++")
add_subdirectory("./mindx_sdk_plugin")
add_subdirectory("./proto")
```
>相比4-1样例中的cmakelist仅添加了`add_subdirectory("./proto")`该行内容

完成后右键cmakelist并reload，执行完毕后在远程环境下可看到已在proto目录下生成相关头文件和库（lib目录中）  
![image.png](img/202107201822.png 'image.png')  

**步骤3** 添加自定义proto的赋值 
本样例中包含的mindx_sdk_plugin基于4-1样例修改得到，包含以下改动：  
1. mindx_sdk_plugin/src/mxpi_sampleplugin/CMakeLists.txt
> 第20行添加了指向自定义proto的编译链接
```cmake
# add custom proto dir
include_directories(${PROJECT_SOURCE_DIR}/../../../proto)
link_directories(${PROJECT_SOURCE_DIR}/../../../proto/lib)
```
>倒数第2行添加了自定义proto库`mxpisampleproto`

2. mindx_sdk_plugin/src/mxpi_sampleplugin/MxpiSamplePlugin.h
>第24行添加了自定义proto相关头文件`#include "mxpiSampleProto.pb.h"`

3. mindx_sdk_plugin/src/mxpi_sampleplugin/MxpiSamplePlugin.cpp
>127-147行为新增一自定义proto结构并赋值后将其加入metadata
```cpp
    // Generate sample proto
    auto mxpiSampleProtoListptr = std::make_shared<sampleproto::MxpiSampleProtoList>();
    auto mxpiSampleProtoptr = mxpiSampleProtoListptr->add_sampleprotovec();

    sampleproto::MxpiMetaHeader* dstMxpiMetaHeaderList = mxpiSampleProtoptr->add_headervec();
    dstMxpiMetaHeaderList->set_datasource(parentName_);
    dstMxpiMetaHeaderList->set_memberid(0);

    mxpiSampleProtoptr->set_intsample(648);
    mxpiSampleProtoptr->set_stringsample("sample proton string with id 648");

    std::string sampleProtoName = "mxpi_sampleproto";
    ret = mxpiMetadataManager.AddProtoMetadata(sampleProtoName, static_pointer_cast<void>(mxpiSampleProtoListptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, sampleProtoName) << "MxpiSamplePlugin add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, sampleProtoName, mxpiErrorInfo);
        return ret;
    }

```
- MxpiMetaHeader结构体的赋值（131-133行）在本样例中非必须，此处注释后不影响实际输出。
- sampleProtoName对应该proto结构在metadata中的名词，与pipeline中outputDataKeys取值对应

**步骤4** 运行测试  
1. C++样例  

>使用菜单项Build/Rebuild All ....后运行该程序，如果无错误可在输出中看到类似的以下字段  
```shell
I0720 18:39:36.202248 31340 main.cpp:129] Results:{"MxpiSampleProto":[{"bytesSample":"","intSample":648,"stringSample":"sample proton string with id 648"}]}
```
2. Python样例  
>修改python样例目录下的run.sh脚本（该脚本用于配置环境变量），确认MX_SDK_HOME环境变量指向正确的位置，自定义插件路径已位于GST_PLUGIN_PATH中，或手动修改环境变量，然后执行程序。正常情况下输出以下类似字段
```shell
{"MxpiSampleProto":[{"bytesSample":"","intSample":648,"stringSample":"sample proton string with id 648"}]}
```
**步骤5** 取出protobuf数据
****
本章节提供修改后的main.cpp和main.py代码原文，位于protoRead文件夹下
****
本步骤讲解如何单独取出自定义的protbuf数据（使用GetProtobuf接口而非GetResult）
- C++样例修改
>在当前clion项目的LD_LIBRARY_PATH环境变量中添加自定义proto库的位置
1. mxVision/C++/CMakeLists.txt
> 第31行添加了指向自定义proto的编译链接
```cmake
# add custom proto dir
include_directories(${PROJECT_SOURCE_DIR}/../../proto)
link_directories(${PROJECT_SOURCE_DIR}/../../proto/lib)
```
>倒数第2行添加了自定义proto库`mxpisampleproto`

2. mxVision/C++/main.cpp
>第20行添加了自定义proto相关头文件`#include "mxpiSampleProto.pb.h"`

>122-130行替换为以下内容
```c++
// choose which metadata to be got.In this case we use the custom "mxpi_sampleproto"
    std::vector<std::string> keyVec;
    keyVec.push_back("mxpi_sampleproto");

    // get stream output
    std::vector<MxStream::MxstProtobufOut> output = mxStreamManager.GetProtobuf(streamName, inPluginId, keyVec);
    if (output.size() == 0) {
        LogError << "output size is 0";
        return APP_ERR_ACL_FAILURE;
    }
    if (output[0].errorCode != APP_ERR_OK) {
        LogError << "GetProtobuf error. errorCode=" << output[0].errorCode;
        return output[0].errorCode;
    }
    LogInfo << "key=" << output[0].messageName;
    LogInfo << "value=" << output[0].messagePtr.get()->DebugString();
```  
>删除145行的`delete output;`

此时输出结果应该类似：  
```shell
I0722 17:58:00.161918 22272 main.cpp:134] key=mxpi_sampleproto
I0722 17:58:00.161969 22272 main.cpp:135] value=sampleProtoVec {
  headerVec {
    dataSource: "mxpi_modelinfer0"
  }
  intSample: 648
  stringSample: "sample proton string with id 648"
}
```
如果需要获取protobuf内部数据项，需要执行强制转换，本样例中添加到140行后：  
```c++
    auto protoResList = std::static_pointer_cast<sampleproto::MxpiSampleProtoList>(output[0].messagePtr);
    LogInfo << "protobuf stringSample=" << protoResList->sampleprotovec(0).stringsample();
```
输出类似`I0722 19:28:03.550653 28737 main.cpp:141] protobuf stringSample=sample proton string with id 648`

- Python样例修改
1. 添加相关库引用
```python
import sys
sys.path.append("../../proto")
import mxpiSampleProto_pb2 as mxpiSampleProto
```

2. 更改数据输入方式为`SendData`而非`SendDataWithUniqueId`（63行），后者不兼容`GetProtobuf`

3. 替换67-72行为以下内容
```python
    # get protobuf with custom
    key_vec = StringVector()
    # choose which metadata to be got.In this case we use the custom "mxpi_sampleproto"
    key_vec.push_back(b"mxpi_sampleproto")
    infer_result = streamManagerApi. \
        GetProtobuf(streamName, inPluginId, key_vec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("infer_result error. \
                errorCode=%d" % (infer_result[0].errorCode))
        exit()

    # print the infer result
    print("GetProtobuf errorCode=%d" % (infer_result[0].errorCode))
    print("KEY: {}".format(str(infer_result[0].messageName)))

    result_protolist = mxpiSampleProto.MxpiSampleProtoList()
    result_protolist.ParseFromString(infer_result[0].messageBuf)
    print("result: {}".format(
        result_protolist.sampleProtoVec[0].stringSample))
```
输出类似于：
```shell
GetProtobuf errorCode=0
KEY: b'mxpi_sampleproto'
result: sample proton string with id 648
```