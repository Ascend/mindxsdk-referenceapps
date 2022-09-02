#! /bin/bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export REPEAT_TUNE=true
atc --model=ErfNet.onnx --output=./ErfNet_bs1 --framework=5 \
    --input_shape="actual_input_1:1,3,512,1024" \
    --soc_version=Ascend310 \
    --input_format=NCHW \
    --output_type=FP32 \
    --insert_op_conf=./erfnet.aippconfig