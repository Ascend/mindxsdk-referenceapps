# /bin/sh
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
atc --input_shape="blob1:8,3,800,1408" --weight="model/count_person.caffe.caffemodel" --input_format=NCHW --output="model/count_person_8.caffe" --soc_version=Ascend310 --insert_op_conf="model/insert_op.cfg" --framework=0 --model="model/count_person.caffe.prototxt"
