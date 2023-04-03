# 模型Tensor数据处理&插件后处理

本章节将指导模型Tensor数据解析和封装相关操作。
配套样例描述如何自定义一个插件进行后处理操作。
****
**注意！**  
本样例中后处理指使用模型输出的原始metadata，自行开发插件来进行后处理。  
当类型为4-2中相关的内置类型时，效率不如后处理so库方式
****

## medadata结构说明
该结构为模型推理插件mxpi_tensorinfer输出的原始数据，按以下层级封装。

1. MxpiTensorPackageList
模型tensor组合列表。
```protobuf
repeated MxpiTensorPackage tensorPackageVec;
```
2. MxpiTensorPackage

模型tensor组合数据结。
```protobuf
repeated MxpiMetaHeader headerVec;
repeated MxpiTensor tensorVec;
```
3. MxpiTensor

模型tensor数据结构。
```protobuf
uint64 tensorDataPtr; // 内存指针数值
int32 tensorDataSize; // 内存大小，需要和实际内存大小一致，否则可能会导致coredump
uint32 deviceId; // Device编号
MxpiMemoryType memType; // 内存类型
uint64 freeFunc; // 内存销毁函数指针
repeated int32 tensorShape; // 张量形状
bytes dataStr; // 内存中的数据
int32 tensorDataType; //内存中张量的数据类型
```

## C++结构体说明
该结构体为TensorBase数据结构相关说明，详情可参考TensorBase.h头文件。  
（位于%SDK%/include/MxBase/Tensor/TensorBase/）  
本处仅摘抄部分关键信息
```c++
enum TensorDataType {
    TENSOR_DTYPE_UNDEFINED = -1,
    TENSOR_DTYPE_FLOAT32 = 0,
    TENSOR_DTYPE_FLOAT16 = 1,
    TENSOR_DTYPE_INT8 = 2,
    TENSOR_DTYPE_INT32 = 3,
    TENSOR_DTYPE_UINT8 = 4,
    TENSOR_DTYPE_INT16 = 6,
    TENSOR_DTYPE_UINT16 = 7,
    TENSOR_DTYPE_UINT32 = 8,
    TENSOR_DTYPE_INT64 = 9,
    TENSOR_DTYPE_UINT64 = 10,
    TENSOR_DTYPE_DOUBLE64 = 11,
    TENSOR_DTYPE_BOOL = 12
};
    // 获取tensor部署的设备类型
    MemoryData::MemoryType GetTensorType() const;
    // buffer记录的数据量
    size_t GetSize() const;
    // buffer 字节数据量
    size_t GetByteSize() const;
    // tensor 的shape
    std::vector<uint32_t> GetShape() const;
    std::vector<uint32_t> GetStrides() const;
    // tensor 的 device
    int32_t GetDeviceId() const;
    // tensor 数据类型
    TensorDataType GetDataType() const;
    uint32_t GetDataTypeSize() const;
    // 判断是否在Host
    bool IsHost() const;
    // 判断是否在Device
    bool IsDevice() const;
    // 获取tensor指针
    void* GetBuffer() const;
    APP_ERROR GetBuffer(void *&ptr, const std::vector<uint32_t> &indices) const;
    // host to device
    APP_ERROR ToDevice(int32_t deviceId);
    // host to dvpp
    APP_ERROR ToDvpp(int32_t deviceId);
    // device to host
    APP_ERROR ToHost();
    // 组batch相关
    static APP_ERROR BatchConcat(const std::vector<TensorBase> &inputs, TensorBase &output);
    static APP_ERROR BatchStack(const std::vector<TensorBase> &inputs, TensorBase &output);
    static APP_ERROR BatchVector(const std::vector<TensorBase> &inputs, TensorBase &output,
        const bool &keepDims = false);
    // 详细信息
    std::string GetDesc();
    // 检查错误
    APP_ERROR CheckTensorValid() const;
```
## 样例说明
参考[4-1插件开发调试指导](4-1插件开发调试指导.md)部署自定义插件样例，示例使用[samplePluginPostProc](../../tutorials/samplePluginPostProc/)作为工程名，远程目录名同样为samplePluginPostProc。  
- 更改mxVision/C++/main.cpp中所使用的pipeline为样例中的SamplePluginPost.pipeline
- 更改mxVision/python/main.py中使用的pipeline为样例中的SamplePluginPost.pipeline
- 相比4-1样例中的SamplePlugin.pipeline，本样例中pipeline使用新后处理框架下的模型推理插件mxpi_tensorinfer输出原始Tensor至自定义插件并完成后处理示例。
>该文件位于样例根目录，但代码中实际指向mxVision/pipeline文件夹下，这是为了与使用原有Sample.pipeline的样例统一目录。实际使用时复制pipeline文件或更改代码中的路径均可  

****
本样例部署测试同4-1，不再重复。
****
**运行结果**
1. C++样例  

>使用菜单项Build/Rebuild All ....后运行该程序，如果无错误可在输出中看到类似的以下字段  其中MxpiSamplePlugin.cpp文件中的日志输出为插件内部结果，main.cpp日志输出为stream结果。此处tensor[0]相关值应该相同
```shell
I0730 10:36:55.459653 11981 MxpiSamplePlugin.cpp:118] MxpiSamplePlugin::Process start
W0730 10:36:55.459954 11981 MxpiSamplePlugin.cpp:98] source Tensor number:3
W0730 10:36:55.460223 11981 MxpiSamplePlugin.cpp:99] Tensor[0] ByteSize in .cpp:172380
I0730 10:36:55.462554 11981 MxpiSamplePlugin.cpp:174] MxpiSamplePlugin::Process end
I0730 10:36:55.462568 11935 main.cpp:129] Results:{"MxpiClass":[{"classId":42,"className":"The shape of tensor[0] in metadata is 172380, Don’t panic!","confidence":0.314}]}
```
2. Python样例  
>修改python样例目录下的run.sh脚本（该脚本用于配置环境变量），确认MX_SDK_HOME环境变量指向正确的位置，自定义插件路径已位于GST_PLUGIN_PATH中，或手动修改环境变量，然后执行程序。正常情况下输出以下类似字段
```shell
{"MxpiClass":[{"classId":42,"className":"The shape of tensor[0] in metadata is 172380, Don’t panic!","confidence":0.314}]}
```