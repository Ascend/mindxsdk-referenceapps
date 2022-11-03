#! /bin/bash
atc --model=PraNet-19.onnx --output=./PraNet-19_bs1 --framework=5 \
    --input_shape="actual_input_1:1,3,352,352" \
    --soc_version=Ascend310 \
    --input_format=NCHW \
    --output_type=FP32 \
    --insert_op_conf=./pranet.aippconfig