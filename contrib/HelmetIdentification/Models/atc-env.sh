#!/bin/bash

# 该脚本用来将onnx模型文件转换成.om模型文件
# This is used to convert onnx model file to .om model file.
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/arm64-linux/atc/ccec_compiler/bin:${install_path}/arm64-linux/atc/bin:$PATH
export PYTHONPATH=${install_path}/arm64-linux/atc/python/site-packages:${install_path}/arm64-linux/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/arm64-linux/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/arm64-linux/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export Home=/home/sd_xiong2/ExampleProject/HelmetIdentification
echo "successfully!!"
# 执行，转换YOLOv5模型
# Execute, transform YOLOv5 model.

atc --model=${Home}/Models/helmet_head_person_s_1.7.0_op11_dbs_sim_t.onnx --framework=5 --output=${Home}/Models/helmet_head_person_s_1.7.0_op11_dbs_sim_t  --insert_op_conf=./aipp_YOLOv5.config --input_format=NCHW --log=info --soc_version=Ascend310 --input_shape="images:1,3,640,640"   # --out_nodes=""
# 说明：out_nodes制定了输出节点的顺序，需要与模型后处理适配。