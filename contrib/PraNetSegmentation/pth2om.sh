
python pranet_pth2onnx.py   ./PraNet-19.pth  ./PraNet-19.onnx
python -m onnxsim PraNet-19.onnx PraNet-19_dybs_sim.onnx --input-shape=1,3,352,352 --dynamic-input-shape
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}
atc --framework=5 --model=PraNet-19_dybs_sim.onnx --output=PraNet-19_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,352,352"  --log=debug --soc_version=Ascend310
