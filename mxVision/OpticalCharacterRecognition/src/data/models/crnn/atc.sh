#!/bin/bash

atc --model=ch_ppocr_server_v2.0_rec_infer_argmax.onnx \
      --framework=5 \
      --input_format=ND \
      --input_shape="x:-1,3,32,-1" \
      --dynamic_dims="1,32;1,64;1,96;1,128;1,160;1,192;1,224;1,256;1,288;1,320;1,352;1,384;1,416;1,448;1,480" \
      --output=crnn_dynamic_dims_16_bs1 \
      --soc_version=Ascend310P3 \
      --log=error

atc --model=ch_ppocr_server_v2.0_rec_infer_argmax.onnx \
      --framework=5 \
      --input_format=ND \
      --input_shape="x:-1,3,32,-1" \
      --dynamic_dims="4,32;4,64;4,96;4,128;4,160;4,192;4,224;4,256;4,288;4,320;4,352;4,384;4,416;4,448;4,480" \
      --output=crnn_dynamic_dims_16_bs4 \
      --soc_version=Ascend310P3 \
      --log=error

atc --model=ch_ppocr_server_v2.0_rec_infer_argmax.onnx \
      --framework=5 \
      --input_format=ND \
      --input_shape="x:-1,3,32,-1" \
      --dynamic_dims="8,32;8,64;8,96;8,128;8,160;8,192;8,224;8,256;8,288;8,320;8,352;8,384;8,416;8,448;8,480" \
      --output=crnn_dynamic_dims_16_bs8 \
      --soc_version=Ascend310P3 \
      --log=error

atc --model=ch_ppocr_server_v2.0_rec_infer_argmax.onnx \
      --framework=5 \
      --input_format=ND \
      --input_shape="x:-1,3,32,-1" \
      --dynamic_dims="16,32;16,64;16,96;16,128;16,160;16,192;16,224;16,256;16,288;16,320;16,352;16,384;16,416;16,448;16,480" \
      --output=crnn_dynamic_dims_16_bs16 \
      --soc_version=Ascend310P3 \
      --log=error

atc --model=ch_ppocr_server_v2.0_rec_infer_argmax.onnx \
      --framework=5 \
      --input_format=ND \
      --input_shape="x:-1,3,32,-1" \
      --dynamic_dims="32,32;32,64;32,96;32,128;32,160;32,192;32,224;32,256;32,288;32,320;32,352;32,384;32,416;32,448;32,480" \
      --output=crnn_dynamic_dims_16_bs32 \
      --soc_version=Ascend310P3 \
      --log=error