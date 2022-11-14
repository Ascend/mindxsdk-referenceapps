#!/bin/bash


# 设置环境变量（请确认install_path路径是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp


# 执行，转换Retinaface模型
# Execute, transform Retinaface model.

atc --framework=5 --model=retinaface.onnx --output=newRetinaface --input_format=NCHW --input_shape="image:1,3,1000,1000" --log=debug --soc_version=Ascend310 --insert_op_conf=../config/aipp.cfg
# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。
