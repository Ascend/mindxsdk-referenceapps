# TensorBase

该类主要有TensorBase数据结构的构造方法和相关的功能接口。

### CreateTensorBase

#### 函数功能

用来创建TensorBase对象，并根据传入参数不同，调用不同的构造函数。

#### 函数原型

```
 template<typename... Param>
static APP_ERROR CreateTensorBase(TensorBase &tensor, Param... params)
{
    tensor = TensorBase(params...);
    auto ret = tensor.CheckTensorValid();
    if (ret != APP_ERR_OK) {
        LogError << "tensor is invalid";
        return ret;
    }
    return APP_ERR_OK;
}
```

#### 参数说明

| 参数名 | 输入/输出 | 说明                                               |
| ------ | --------- | -------------------------------------------------- |
| tensor | 输出      | TensorBase对象，用于接收构造的TensorBase对象       |
| params | 输入      | 用于构造TensorBase的参数，具体参数可参考构造函数。 |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

### TensorBase

#### 函数功能

tensorBase的构造函数，用来创建TensorBase对象，根据传入的不同参数可选用不同的构造函数。

**函数原型1**

```c++
TensorBase(const MemoryData &memoryData, const bool &isBorrowed, const std::vector<uint32_t> &shape, const TensorDataType &type);
```

**函数原型2**

```c++
TensorBase(const std::vector<uint32_t> &shape, const TensorDataType &type, const MemoryData::MemoryType &bufferType, const int32_t &deviceId);
```

**函数原型3**

```c++
TensorBase(const std::vector<uint32_t> &shape, const TensorDataType &type, const int32_t &deviceId);
```

**函数原型4**

```
TensorBase(const std::vector<uint32_t> &shape, const TensorDataType &type);
```

**函数原型5**

```c++
TensorBase(const std::vector<uint32_t> &shape);
```

#### 参数说明

| 参数名     | 输入/输出 | 说明                                                         |
| ---------- | --------- | ------------------------------------------------------------ |
| memoryData | 输入      | 用于构造TensorBase对象的参数，内存管理结构体，可参看11.2.1.10 MemoryData |
| isBorrowed | 输入      | 表示传入的MemoryData数据是否要Tensor主动释放；若为true，表示不需要主动释放，用户自行释放；若为false，用户不需要手动释放，Tensor析构时自动释放 |
| shape      | 输入      | 用于构造TensorBase对象的参数，张量的形状                     |
| type       | 输入      | 用于构造TensorBase对象的参数，TensorDataType类型数据，可参看TensorDataType枚举说明 |
| bufferType | 输入      | 用于构造TensorBase对象的参数，张量数据的内存类型             |
| deviceId   | 输入      | 用于构造TensorBase对象的参数，int类型数据，设备编号          |

**注意：** **函数原型2、3、4、5**只是预先设置了shape，但不会申请内存空间，需要调用**TensorBaseMalloc**才能申请对应内存空间。**函数原型1**使用场景是用户在外部申请好内存空间，构造Tensor对象的时候直接引用申请好的内存，要保证空间大小与Tensor对象的shape要一致，同时可以选择外部申请的空间是Tensor对象内部自行释放还是用户外部释放。

##### TensorDataType枚举说明

```
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
```

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

### TensorBaseMalloc

#### 函数功能

用于获取TensorBase对象的内存。没有提供释放内存的函数，原因是Tensor对象在析构时自动释放内存

#### 函数原型

```
static APP_ERROR TensorBaseMalloc(TensorBase &tensor);
```

#### 参数说明

| 参数名 | 输入/输出 | 说明                         |
| ------ | --------- | ---------------------------- |
| tensor | 输入      | 用来获取对应内存的tensor数据 |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

### TensorBaseCopy

#### 函数功能

内存复制函数，根据MemoryData中指定的内存位置在Host侧和Device侧之间进行复制。

#### 函数原型
```
static APP_ERROR TensorBaseCopy(TensorBase &dst, const TensorBase &src);
```

#### 参数说明

| 参数名 | 输入/输出 | 说明               |
| ------ | --------- | ------------------ |
| dst    | 输出      | 复制后的目标内存。 |
| src    | 输入      | 待复制的源内存。   |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

### GetTensorType

#### 函数功能

用于获取tensor的内存类型。

#### 函数原型

```
MemoryData::MemoryType GetTensorType() const;
```

#### 返回参数说明

| 数据结构   | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| MemoryType | 申请的内存类型：● MEMORY_HOST对应Host侧 ● MEMORY_DEVICE对应Device侧 ● MEMORY_DVPP对应DVPP侧 ● MEMORY_HOST_MALLOC对应malloc申请内存  ● MEMORY_HOST_NEW对应new申请内存 |

##### MemoryType枚举说明

```
enum MemoryType {
 MEMORY_HOST,
 MEMORY_DEVICE,
 MEMORY_DVPP,
 MEMORY_HOST_MALLOC,
 MEMORY_HOST_NEW,
 }
```

### GetSize

#### 函数功能

用于获取张量数据对应内存大小。size大小应与实际内存大小一 致，否则可能会导致程序 coredump。

#### 函数原型

```
size_t GetSize() const;
```

#### 返回参数说明

| 数据结构 | 说明         |
| -------- | ------------ |
| size_t   | 返回具体数值 |

#### 功能解析

```
tensor [224, 224, 3]
tensor.GetSize() -> 224*224*3
```

### GetByteSize

#### 函数功能

用于获取buffer 字节数据量。

#### 函数原型

```
size_t GetByteSize() const;
```

#### 返回参数说明

| 数据结构 | 说明         |
| -------- | ------------ |
| size_t   | 返回具体数值 |

#### 功能解析

```
tensor [224, 224, 3]
tensor.GetByteSize() -> 224*224*3*dtype(TensorDataType属性对应的字节数)
比如：TensorDataType为TENSOR_DTYPE_FLOAT16
float16为2个字节长度因此
tensor.GetByteSize() -> 224*224*3*2
```

### GetShape

#### 函数功能

用于获取tensor 的形状。

#### 函数原型

```
std::vector<uint32_t> GetShape() const;
```

#### 返回参数说明

| 数据结构              | 说明       |
| --------------------- | ---------- |
| std::vector<uint32_t> | shape数组 |

#### 函数功能

用于获取tensor 的步长。

#### 函数原型

```
std::vector<uint32_t> GetStrides() const;
```

#### 返回参数说明

| 数据结构              | 说明       |
| --------------------- | ---------- |
| std::vector<uint32_t> | 是一个数组 |

#### 功能解析

```
tensor:(1,2,3,4)
tensor.GetStrides() -> strides:(24,12,4,1)   
解析：strides:(2*3*4,3*4,4,1)
```

### GetDeviceId

#### 函数功能

用于获取tensor 的设备号。

#### 函数原型

```
int32_t GetDeviceId() const;
```

#### 返回参数说明

| 数据结构 | 说明                        |
| -------- | --------------------------- |
| int32_t  | 返回设备号，是一个int类型。 |

### GetDataType

#### 函数功能

用于获取内存中张量的数据类型。

#### 函数原型

```
TensorDataType GetDataType() const;
```

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| TensorDataType | 请参考TensorDataType枚举说明。 |

### GetDataTypeSize
#### 函数功能
用于获取tensor的 数据类型。根据TensorDataType的具体类型返回该类型字节长度。
#### 函数原型
```
uint32_t GetDataTypeSize() const;
```

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| uint32_t | 是一个int类型的值。 |

##### TensorDataType和GetDataTypeSize对应值

```
TENSOR_DTYPE_UNDEFINED: "indefined" -> ZERO_BYTE = 0 ;
TENSOR_DTYPE_FLOAT32: "float32" -> FOUR_BYTE = 4 ;
TENSOR_DTYPE_FLOAT16: "float16" -> TWO_BYTE = 2 ;
TENSOR_DTYPE_INT8: "int8" -> ONE_BYTE = 1 ;
TENSOR_DTYPE_INT32: "int32" -> FOUR_BYTE = 4 ;
TENSOR_DTYPE_UINT8: "uint8" -> ONE_BYTE = 1 ;
TENSOR_DTYPE_INT16: "int16" -> TWO_BYTE = 2 ;
TENSOR_DTYPE_UINT16: "uint16" -> TWO_BYTE = 2 ;
TENSOR_DTYPE_UINT32: "uint32" -> FOUR_BYTE = 4 ;
TENSOR_DTYPE_INT64: "int64" -> EIGHT_BYTE = 8 ;
TENSOR_DTYPE_UINT64: "uint64" -> EIGHT_BYTE = 8 ;
TENSOR_DTYPE_DOUBLE64: "double64" -> EIGHT_BYTE = 8 ;
TENSOR_DTYPE_BOOL: "bool" -> ONE_BYTE = 1 ;
```

#### 返回参数说明

### IsHost

#### 函数功能
用于判断tensor对象是否在Host侧。
#### 函数原型
```
bool IsHost() const;
```

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| bool | 布尔型，值只有 真 （true) 和假 （false）。 |

### IsDevice
#### 函数功能
用于判断tensor对象是否在Device侧。
#### 函数原型
```
bool IsDevice() const;
```

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| bool | 布尔型，值只有 真 （true) 和假 （false）。 |

### GetBuffer（举例）
#### 函数功能
用于获取tensor指针

#### 函数原型1

```
void* GetBuffer() const;
```

#### 返回参数说明

| 数据结构 | 说明                       |
| -------- | -------------------------- |
| void*    | 指针的首地址。即buffer指针 |

#### 功能解析

```
tensor.GetBuffer()  ->  buffer
```

#### 函数原型2

```
APP_ERROR GetBuffer(void *&ptr, const std::vector<uint32_t> &indices) const;
```

#### 参数说明

| 参数名  | 输入/输出 | 说明                          |
| ------- | --------- | ----------------------------- |
| ptr     | 输出      | 获取到的buffer指针。          |
| indices | 输入      | 传入的索引列表,待复制的源内存 |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

#### 功能解析

```
uint32_t offset = 0;
strides = GetStrides();
for (uint32_t i  = 0; i < indices.size(); i++){
	offset += indices[i] * strides[i];
}
ptr = (void*)((uint8_t*)GetBuffer() + offset);
# 解析 其本质是调用的GetBuffer()函数原型1
```

#### 函数原型3

```
template<typename T, typename... Ix>
APP_ERROR GetBuffer(T* &value, Ix... index) const
{
	std::vector<uint32_t> indices = {};
	// GetIndices把索引index，push_back进indices
	GetIndices(indices, index...);
	void *ptr = nullptr;
	APP_ERROR ret = GetBuffer(ptr, indices);
	if (ret != APP_ERR_OK) {
		LogError << GetError(ret) << "GetBuffer failed.";
		return ret;
	}
	value = (T*)ptr;
	return APP_ERR_OK;
}
```

#### 参数说明

| 参数名 | 输入/输出 | 说明             |
| ------ | --------- | ---------------- |
| value  | 输出      | 获取到的buffer。 |
| index  | 输入      | 传入的索引值     |

#### 功能解析

```
如代码所示，将传入的index参数通过GetIndices函数push_back进indices，再调用GetBuffer函数原型2 GetBuffer(ptr, indices)，函数原型2再调用函数原型1，其本质是调用GetBuffer函数原型1。
```

### GetValue

#### 函数功能

获取Host侧的tensor的buffer，其中判断了tensor是否在host，参数类型。然后在其中调用GetBuffer函数实现获取buffer的功能。

#### 函数原型
```
template<typename T, typename... Ix>
APP_ERROR GetValue(T &value, Ix... index) const
{
	if (!IsHost()) {
		LogError << "this tensor is not in host. you should deploy it to host";
		return APP_ERR_COMM_FAILURE;
	}
	if (sizeof(T) != GetDataTypeSize()) {
		LogError << "output date type is not match to tensor date type(" << GetDataType() << ")";
		return APP_ERR_COMM_FAILURE;
	}
	T *ptr = nullptr;
	APP_ERROR ret = GetBuffer(ptr, index...);
	if (ret != APP_ERR_OK) {
		LogError << GetError(ret) << "GetBuffer failed.";
        return ret;
    }
    value = *ptr;
	return APP_ERR_OK;
}
```

#### 参数说明

| 参数名 | 输入/输出 | 说明               |
| ------ | --------- | ------------------ |
| value | 输出  | 获取到对应tensor的buffer |
| index | 输入      | 待获取tensor的索引 |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |


### ToDevice
#### 函数功能
将tensor对象部署到device侧内存。
#### 函数原型
```
APP_ERROR ToDevice(int32_t deviceId);
```

#### 参数说明

| 参数名 | 输入/输出 | 说明               |
| ------ | --------- | ------------------ |
| deviceId    | 输入 | 设备号 |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

### ToDvpp
#### 函数功能
将tensor对象部署到dvpp内存中。
#### 函数原型
```
APP_ERROR ToDvpp(int32_t deviceId);
```

#### 参数说明

| 参数名 | 输入/输出 | 说明               |
| ------ | --------- | ------------------ |
| deviceId    | 输入 | 设备号 |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

### ToHost
#### 函数功能
将tensor对象部署到 host侧内存。 
#### 函数原型
```
APP_ERROR ToHost();
```

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

### BatchConcat
#### 函数功能

将多batch的tensor组合成一个tensor

#### 函数原型
```
static APP_ERROR BatchConcat(const std::vector<TensorBase> &inputs, TensorBase &output);
```

#### 参数说明

| 参数名 | 输入/输出 | 说明               |
| ------ | --------- | ------------------ |
| inputs    |    输入   | tensor列表 |
| output    |    输出   | 组合后的tensor对象 |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

#### 功能解析

```
tensor1 [1, 224, 224, 3]
tensor2 [1, 224, 224, 3]
tensor3 [1, 224, 224, 3]
BatchConcat([tensor1, tensor2, tensor3]) -> tensor[3, 224, 224, 3]
```

### BatchStack

#### 函数功能

将多batch的tensor组合成一个tensor，相比于BatchConcat拓展了tensor的维度

#### 函数原型
```
static APP_ERROR BatchStack(const std::vector<TensorBase> &inputs, TensorBase &output);
```

#### 参数说明

| 参数名 | 输入/输出 | 说明               |
| ------ | --------- | ------------------ |
| inputs    |    输入   | tensor列表 |
| output    |    输出   | 组合后的tensor对象 |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

#### 功能解析

```
tensor1 [224, 224, 3]
tensor2 [224, 224, 3]
tensor3 [224, 224, 3]
BatchStack([tensor1, tensor2, tensor3]) -> tensor[3, 224, 224, 3]
```

### BatchVector

#### 函数功能
组batch，将多batch的tensor组合成一个tensor。该函数实际是调用BatchConcat和BatchStack实现组batch操作，通过keepDims参数控制具体调用函数。
#### 函数原型
```
static APP_ERROR BatchVector(const std::vector<TensorBase> &inputs, TensorBase &output, const bool &keepDims = false);
```

#### 参数说明

| 参数名 | 输入/输出 | 说明               |
| ------ | --------- | ------------------ |
| inputs    |    输入   | 多batch模型，将TensorBase对象列表作为输出 |
| output    |    输出   | 组batch后输出一个TensorBase对象 |
| keepDims  |  输入   | 保持其多维特性的参数，true即保持其维度特性函数内部调用函数BatchConcat实现，false即不保持维度特性函数内部调用函数BatchStack实现 |

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |

### GetDesc
#### 函数功能
用于获取详细信息
#### 函数原型
```
std::string GetDesc();
```

#### 返回参数说明

| 数据结构    | 说明     |
| ----------- | -------- |
| std::string | 详情信息 |

### CheckTensorValid
#### 函数功能
用于查看tensor对象是否合法
#### 函数原型
```
APP_ERROR CheckTensorValid() const;
```

#### 返回参数说明

| 数据结构  | 说明              |
| --------- | ----------------- |
| APP_ERROR | 请参考APP_ERROR。 |
