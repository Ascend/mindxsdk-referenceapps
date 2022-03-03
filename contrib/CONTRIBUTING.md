### 介绍

mindxsdk-referenceapps欢迎各位开发者的加入，希望各位开发者遵循社区的行为准则，共同建立一个开放和受欢迎的社区 [ Ascend社区行为准则 1.0 版本]([code-of-conduct_zh_cn.md · Ascend/community - 码云 - 开源中国 (gitee.com)](https://gitee.com/ascend/community/blob/master/code-of-conduct_zh_cn.md))

### 贡献要求

请贡献者在提交代码之前签署CLA协议，“个人签署”，[链接](https://clasign.osinfra.cn/sign/Z2l0ZWUlMkZhc2NlbmQ=)

如您完成签署，可在自己提交的PR评论区输入/check-cla进行核实校验

开发者提交的内容包括项目源码、配置文件、readme、启动脚本等文件，并遵循以下标准提交：

### 一、提交内容

- **提交清单**

| 文件         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **README**   | 包含第三方依赖安装、模型转换、编译、运行指导等内容，能指导端到端使用 |
| 代码         | 包含插件开发的C++代码、CMakeLists.txt、python/C++推理运行代码、精度与性能测试代码 |
| 配置文件     | 运行时相关配置文件，用于加载相关运行参数的文件               |
| pipeline文件 | MindX SDK的编排文件                                          |
| 启动脚本     | 包括编译、运行、测试、模型转换等脚本                         |
| 参考设计案例 | 参考设计案例文档或案例视频                                   |

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

### 二、源码

1、MindX SDK离线推理请使用`C++`或`python`代码，符合第四部分编码规范

2、贡献者参考设计代码目录命名规则

```shell
mindxsdk-referenceapps/contrib/参考设计名称(英文)
```

### 三、License规则

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

### 四、编程规范

- 规范标准

  C++遵循Google编程规范，Python代码均遵循PEP 8编码规范

  规范参考链接：[zh-cn/contribute/OpenHarmony-cpp-coding-style-guide.md · OpenHarmony/docs - Gitee.com](https://gitee.com/openharmony/docs/blob/master/zh-cn/contribute/OpenHarmony-cpp-coding-style-guide.md)

- 规范备注（前4条规则C++适用）

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

### 五、代码提交规范

- 关键要求：

1、请将**`mindxsdk-referenceapps`**仓**fork**到个人分支，基于个人分支提交代码到个人**fork仓**，并创建**`Pull Requests`**，提交合并请求到主仓上

**参考Fork+Pull Requests 模式**：https://gitee.com/help/articles/4128#article-header0

> pr提交后请不要再关闭pr，一切操作都在不关pr的条件下进行操作

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

### 六、ISSUE提交规范

1、ISSUE提交内容需包含三部分：当前行为、预期行为、复现步骤

2、ISSUE提交模板：

```
一、问题现象（附报错日志上下文）：
### 当前现象
    xxxx
    
### 预期现象
    xxxx

二、软件版本:
-- CANN 版本 (e.g., CANN 3.0.x，5.x.x):  
--Tensorflow/Pytorch/MindSpore 版本:
--Python 版本 (e.g., Python 3.7.5):
-- MindStudio版本 (e.g., MindStudio 2.0.0 (beta3)):
--操作系统版本 (e.g., Ubuntu 18.04):

三、复现步骤：
xxxx


四、日志信息:
xxxx
### 请根据自己的运行环境参考以下方式搜集日志信息，如果涉及到算子开发相关的问题，建议也提供UT/ST测试和单算子集成测试相关的日志。

日志提供方式:
### 将日志打包后作为附件上传。若日志大小超出附件限制，则可上传至外部网盘后提供链接。

### 获取方法请参考wiki：
https://gitee.com/ascend/modelzoo/wikis/%E5%A6%82%E4%BD%95%E8%8E%B7%E5%8F%96%E6%97%A5%E5%BF%97%E5%92%8C%E8%AE%A1%E7%AE%97%E5%9B%BE?sort_id=4097825
```

