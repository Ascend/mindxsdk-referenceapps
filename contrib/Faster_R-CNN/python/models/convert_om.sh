#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_AIR_PATH] [AIPP_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.air ./aipp.cfg xx"

  exit 1
fi

input_air_path=$1
aipp_cfg_file=$2
output_om_path=$3

echo "Input AIR file path: ${input_air_path}"
echo "Output OM file path: ${output_om_path}"

atc --input_format=NCHW \
    --framework=1 \
    --model="${input_air_path}" \
    --input_shape="x:1, 3, 768, 1280; im_info: 1, 4" \
    --output="${output_om_path}" \
    --insert_op_conf="${aipp_cfg_file}" \
    --precision_mode=allow_fp32_to_fp16 \
    --soc_version=Ascend310 \
    --op_select_implmode=high_precision