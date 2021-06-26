### 介绍

MindX SDK, 欢迎各位开发者

### 贡献要求

开发者提交的模型包括源码、readme、参考模型license文件、测试用例和readme，并遵循以下标准

请贡献者在提交代码之前签署CLA协议，“个人签署”，[链接](https://clasign.osinfra.cn/sign/Z2l0ZWUlMkZhc2NlbmQ=)

如您完成签署，可在自己提交的PR评论区输入/check-cla进行核实校验

### 一、源码

1、MindX SDK离线推理请使用`C++`或`python`代码，符合第四部分编码规范

2、贡献者参考设计代码目录命名规则

```shell
mindxsdk-referenceapps/contrib/参考设计名称(英文)
```

### 二、License规则

涉及的代码、启动脚本都均需要在开始位置添加华为公司 License [华为公司 License链接](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)

- **C++**

```c++
/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

- **python**&**shell**

```python
# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

> 关于License声明时间，应注意： 2021年新建的文件，应该是Copyright 2021 Huawei Technologies Co., Ltd 2020年创建年份，2020年修改年份，应该是Copyright 2020 Huawei Technologies Co., Ltd

### 三、提交内容

- **提交清单**

| 文件         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **README**   | 包含第三方依赖安装、模型转换、编译、运行指导等内容，能指导端到端使用 |
| 代码         | 包含插件的C++代码、CMakeLists.txt、python/C++推理运行代码    |
| 配置文件     | 运行时相关配置文件，用于加载相关运行参数的文件               |
| pipeline文件 | MindX SDK的编排文件                                          |
| 启动脚本     | 包括编译、运行、测试、模型转换等脚                           |

- **典型的目录结构**

```bash

├── config #配置文件目录
│   └── configure.cfg
├── model #模型目录
├── pipeline
│   └── test.pipeline
├── Plugin1 #插件1工程目录
│   ├── CMakeLists.txt
│   ├── Plugin1.cpp
│   └── Plugin1.h
├── Plugin2 #插件2工程目录
│   ├── CMakeLists.txt
│   ├── Plugin2.cpp
│   └── Plugin2.h
├── main.cpp
├── main.py
├── README.md
├── build.sh
└── run.sh

```

**注意**：相关输入的数据（图像、视频等）请不要上传到代码上，请在README注明如何获取

### 四、代码提交规范

- 关键要求：

1、请将**`mindxsdk-referenceapps`**仓**fork**到个人分支，基于个人分支提交代码到个人**fork仓**，并创建**`Pull Requests`**，提交合并请求到主仓上

**参考Fork+Pull Requests 模式**：https://gitee.com/help/articles/4128#article-header0

2、PR标题模板

```
 [xxx学校] [xxx参考设计]
```

3、PR内容模板

```
### 相关的Issue

### 原因（目的、解决的问题等）

### 描述（做了什么，变更了什么）

### 测试用例（新增、改动、可能影响的功能）
```

### 五、编程规范

- 规范标准

1、C++代码遵循google编程规范：Google C++ Coding Guidelines；单元测测试遵循规范： Googletest Primer。

2、Python代码遵循PEP8规范：Python PEP 8 Coding Style；单元测试遵循规范： pytest

- 规范备注

1、优先使用string类型，避免使用char*；

2、禁止使用printf，一律使用cout；

3、内存管理尽量使用智能指针；

4、不准在函数里调用exit；

5、禁止使用IDE等工具自动生成代码；

6、控制第三方库依赖，如果引入第三方依赖，则需要提供第三方依赖安装和使用指导书；

7、一律使用英文注释，注释率30%--40%，鼓励自注释；

8、函数头必须有注释，说明函数作用，入参、出参；

9、统一错误码，通过错误码可以确认那个分支返回错误；

10、禁止出现打印一堆无影响的错误级别的日志；