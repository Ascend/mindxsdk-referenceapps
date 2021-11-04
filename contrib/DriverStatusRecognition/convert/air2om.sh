model_path=$1
output_model_name=$2
cfg=$3
atc --model=$model_path \
    --framework=1 \
    --output=$output_model_name \
    --input_format=NCHW --input_shape="x:1,3,224,224" \
    --enable_small_channel=1 \
    --log=error \
    --soc_version=Ascend310 \
    --insert_op_conf=$cfg 
    
