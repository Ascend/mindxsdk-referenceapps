#!/bin/bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export REPEAT_TUNE=true
python ErfNet_pth2onnx.py erfnet_pretrained.pth ErfNet_origin.onnx
python modify_bn_weights.py ErfNet_origin.onnx ErfNet.onnx
atc --framework=5 --model=ErfNet.onnx --output=ErfNet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,512,1024" --log=debug --soc_version=Ascend310
if [ -f "ErfNet_bs1.om" ] ; then
    echo "success"
else
    echo "fail!"
fi
