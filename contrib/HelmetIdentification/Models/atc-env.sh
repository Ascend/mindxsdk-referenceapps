#!/bin/bash

# This is used to convert onnx model file to .om model file.
. /usr/local/Ascend/ascend-toolkit/set_env.sh   # The path where Ascend-cann-toolkit is located
export Home="./path/"
# Home is set to the path where the model is located

# Execute, transform YOLOv5 model.
atc --model="${Home}"/YOLOv5_s.onnx --framework=5 --output="${Home}"/YOLOv5_s  --insert_op_conf=./aipp_YOLOv5.config --input_format=NCHW --log=info --soc_version=Ascend310 --input_shape="images:1,3,640,640" 
# --model is the path where onnx is located. --output is the path where the output of the converted model is located