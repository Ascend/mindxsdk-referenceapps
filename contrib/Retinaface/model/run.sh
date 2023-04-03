#!/bin/bash


# 执行，转换Retinaface模型
# Execute, transform Retinaface model.

atc --framework=5 --model=retinaface.onnx --output=newRetinaface --input_format=NCHW --input_shape="image:1,3,1000,1000" --log=debug --soc_version=Ascend310 --insert_op_conf=../config/aipp.cfg
# 说明1：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
# 说明2：若用例执行在310B上，则--soc_version=Ascend310需修改为Ascend310B1