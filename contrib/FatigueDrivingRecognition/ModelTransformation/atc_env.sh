#!/bin/bash

# This is used to convert onnx model file to .om model file.
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export Home="./path"
# Home is set to the path where the model is located

# Execute, transform PFLD model.
atc --framework=5 --model="${Home}"/pfld_106.onnx --output="${Home}"/pfld_106 --input_format=NCHW --insert_op_conf=./aipp_pfld_112_112.aippconfig  --input_shape="input_1:1,3,112,112" --log=debug --soc_version=Ascend310
# --model is the path where onnx is located. 
# --output is the path where the output of the converted model is located