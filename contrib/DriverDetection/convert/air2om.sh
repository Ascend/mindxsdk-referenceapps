model_path=$1
output_model_name=$2

#/usr/local/Ascend/atc/bin/atc \
atc --model=$model_path \
    --framework=1 \
    --output=$output_model_name \
    --input_format=NCHW --input_shape="x:1,3,224,224" \
    --enable_small_channel=1 \
    --log=error \
    --soc_version=Ascend310 \
    --insert_op_conf=./yuv_aipp.config 
    #--input_fp16_nodes=x \
    # --output_type=FP32
