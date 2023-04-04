#! /bin/bash
atc --model=ErfNet.onnx --output=./ErfNet_bs1 --framework=5 \
    --input_shape="actual_input_1:1,3,512,1024" \
    --soc_version=Ascend310B1 \
    --input_format=NCHW \
    --output_type=FP32 \
    --insert_op_conf=./erfnet.aippconfig