# 挂载自定义proto结构体开发指导

本章节将指导在SDK自带的图像分类识别样例中挂载自定义proto结构并输出的流程。  
在执行本样例前，应已成功远程部署并运行4-1章节的自定义插件样例。  
[点击跳转代码样例](https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/tutorials/SamplePostProcess)

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
- 相比4-1样例中的SamplePlugin.pipeline仅修改了51行为`"outputDataKeys": "mxpi_sampleproto"`，使输出的结果为自定义的proto结构
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
使用菜单项Build/Rebuild All ....后运行该程序，如果无错误可在输出中看到l类似的以下字段  
```shell
I0720 18:39:36.202248 31340 main.cpp:129] Results:{"MxpiSampleProto":[{"bytesSample":"","intSample":648,"stringSample":"sample proton string with id 648"}]}
```