# 模型精度评估工具

## 1 内容简介

本样例是由昇腾系列Atlas服务器搭载，用于AI模型精度评估的实用工具。

此工具提供以下功能：

1. 指定数据集和模型，进行自动推理，收集结果；
2. 指定推理结果和评估标准，进行精度评估。

此工具具有以下特点：

1. 简易部署：部署Atlas服务器环境后，只需要根据模型文件的格式安装对应的推理框架即可使用。
2. 应用场景广泛：此工具当前已支持多种框架下的模型推理，包括：MindSpore, Caffe, TensorFlow, PyTorch, 以及基于MindX SDK的离线推理框架；另外，此工具还支持多种数据集的自动处理和结果自动评估，包括：COCO, SVT等。
3. 可扩展性：此工具提供简易的开发方式，用户可自行开发包括但不限于：
   1. 其他数据集的处理与评估方法；
   2. 基于其他框架的推理方法。



## 2 环境依赖

- 支持的硬件形态和操作系统版本

| 硬件形态                              | 操作系统版本   |
| ------------------------------------- | -------------- |
| x86_64+Atlas 300I 推理卡（型号3010）  | Ubuntu 18.04.1 |
| x86_64+Atlas 300I 推理卡 （型号3010） | CentOS 7.6     |
| ARM+Atlas 300I 推理卡 （型号3000）    | Ubuntu 18.04.1 |
| ARM+Atlas 300I 推理卡 （型号3000）    | CentOS 7.6     |

- 软件依赖

| 软件名称    | 版本  |
| ----------- | ----- |
| pycocotools | 2.0   |
| MindX SDK   | 2.0.1 |
| Python      | 3.7.5 |



## 3 项目目录

本代码仓名称为precision_analysis，工程目录如下图所示：

```
├── precision_analysis
│   ├── executor
│   |   ├── data
|   |   |   ├── dataloader.py
|   |   |   └── image_loader.py
│   |   ├── model
|   |   |   ├── base.py
|   |   |   ├── caffe_model.py
|   |   |   ├── mindspore_model.py
|   |   |   ├── onnx_model.py
|   |   |   ├── pb_model.py
|   |   |   └── pipeline.py
│   |   ├── element.py
│   |   ├── inference.py
│   |   └── postprocess.py
│   ├── indicator
│   |   ├── criterion.py
│   |   └── metrics.py
│   ├── interface
│   |   ├── compare.py
│   |   ├── eval.py
│   |   └── summary.py
│   ├── test
│   |   ├── build.py
│   |   ├── test_CRNN.py
│   |   ├── test_Metrics.py
│   |   └── test_ssd_mobilenet_fpn.py
│   ├── utils
│   |   ├── checker.py
│   |   ├── coding_conversion.py
│   |   ├── collection.py
│   |   ├── arguments.py
│   |   ├── constants.py
│   |   └── parser.py
│   ├── main.py
│   ├── test.py
│   ├── make_boarder_test.py
│   ├── README.md
│   └── requirements.txt
```



## 4 使用方法与自定义开发

通常情况下的使用方法可参考 main.py ，以下为各模块功能具体介绍。

此工具包含以下功能模块：

1. ModelExecutor：覆盖多个框架推理功能的基类，另外还可完成基于MindX SDK的推理任务。
2. DataLoader：提供数据集自动化处理功能，将数据集进行格式转换和适当预处理后，送入推理流程，同样作为流程化计算模块而存在；并且，该模块可同时生成数据集信息参数列表（命名为 shared_params）,并传入后续流程，用户也可以根据自己的数据集生成相关参数，利用shared_params进行传递（常见参数有：图片宽高、图片格式等）。
3. InferenceExecutor：可灵活使用以上三个流程化计算模块，并采用流程编排的方式组合而成。是多个计算流程的串行集合，并整体作为推理服务模块的提供者，供精度评估过程使用。
4. EvalCriterion：提供多个常见的精度指标的计算方法。
5. Evaluator：对外提供精度评估接口，对内通过调用InferenceExecutor完成推理计算，再通过调用EvalCriterion完成精度评估。
6. PipeElement：可自定义函数继承该基类，并作为计算方法在InferenceExecutor中使用

以上所有模块均采用自定义开发方式，已提供多种常见方法，用户如需其他功能，可参考已有方法自行开发。



## 5 运行样例

此目录已包含 SSD_MobileNet_FPN 与 CRNN 两个模型的多个运行样例，这里以 SSD_MobileNet_FPN 的运行流程作为样例介绍。

1. 运行 

   `python main.py --mode test.ssd_mobilenet_fpn.pipeline -data-loading-path ${coco数据集路径} -label-loading-path ${coco数据集标签路径} -pipeline-cfg-path ${SDK_pipeline文件路径}  -stream-name ${pipeline配置stream名称}`

   可通过配置 --mode 参数的不同值选择不同的运行样例（当前仅提供 test 运行模式），其中包括（可参考 utils.constants）：

   1. test.ssd_mobilenet_fpn.pipeline 为运行 MindX SDK pipeline 推理样例
   2. test.ssd_mobilenet_fpn.inference 为运行 推理流程模块 的样例
   3. test.ssd_mobilenet_fpn.evaluation 为运行 模型精度评估 的样例

   配置 ${coco数据集路径}：下载coco数据集，配置数据集图片路径，例如：./coco/val2017

   配置 ${coco数据集标签路径}：配置数据集标签文件路径：例如：./coco/annotations/instances_val2017.json

   配置 ${SDK_pipeline文件路径}：运行的pipeline的存放路径

   配置 ${pipeline配置stream名称}：运行的pipeline中的stream名称

2. 该函数调用 test 目录下的 test_ssd_mobilenet_fpn.py 中的 test_pipeline 功能得到样例运行结果

3. 可修改 ssd_mobilenet_fpn 为 crnn 来运行 CRNN 模型的样例



