# Individual attribute recognition

## 介绍

行业SDK流程编排，该项目主要实现了从Sample.pipeline中读取pipeline创建推理stream，然后读取一张图片送到stream进行推理，获取推理结构后把结果打印出来，最后销毁stream。
为了评测，可以从Sample_proto.pipline中读取pipeline创建推理stream，然后读入评测文档，根据评测文档中的数据集路径读入图片，最后将推理结果输出到一个txt文档中，通过对比脚本
可以得到评测的结果。

## 配置

确保pipeline所需要的yolov4和resnet50模型文件在'../models'中存在，run.sh脚本中LD_LIBRARY_PATH设置了ACL动态库链接路径为/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64，
如果实际环境中路径不一致，需要替换为实际的目录。
本项目涵盖人脸识别，人脸关键点提取，人脸对齐，人脸属性识别模型整体的代码。
main.py以及Sample.pipeline是本项目的核心内容，完成人脸识别，人脸关键点提取，人脸对齐以及人脸属性识别整体的流程。在main.py中修改输入图像的路径即可得到相应的推理结果。
项目目录下提供了三组可供测试使用的test1,test2,test3。如有需要，可以修改Sample.pipeline完成更丰富的功能。
项目评测部分主要由在数据集CelebA上完成。该部分由attr_main.py、cal_accuracy.py、Sample_proto.pipeline和groudtruth组成。将数据集文件按照text_full中的路径存放到相应位置，运行脚本文件，可以得到
一个txt输出，再通过运行cal_accuracy.py完成模型输出与标准值的对比，可以查看到评测结果。


## 运行

```python3.7/bash
运行项目：bash ./run.sh 或者 python3.7 main.py
运行评测：python3.7 attr_main.py
python3.7 cal_accuracy.py --gt-file=./test_full.txt --pred-file=./img_result.txt
```
如果使用过程中遇到问题，请联系华为技术支持。