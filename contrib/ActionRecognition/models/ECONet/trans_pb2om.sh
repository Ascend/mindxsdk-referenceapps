#!/bin/bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export SLOG_PRINT_TO_STDOUT=1

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run.sh" ; exit ; } ; pwd)

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc \
 --mode=0 \
 --model="${CUR_PATH}/ucf101_best.pb" \
 --output="${CUR_PATH}/ECONet" \
 --soc_version=Ascend310 \
 --input_format=NHWC\
 --input_shape="clip_holder:8,224,224,3" \
 --insert_op_conf="${CUR_PATH}/eco_aipp.cfg" \
 --framework=3 \
 log=debug > log.txt
