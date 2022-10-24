#!/bin/bash

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}

atc --model=./jester.onnx --framework=5 --output=./jester --input_format=NCDHW  --soc_version=Ascend310  --precision_mode=allow_fp32_to_fp16 --op_select_implmode=high_precision

if [ -f "./jester.om" ]; then
    echo "success"
else
    echo "fail!"
fi