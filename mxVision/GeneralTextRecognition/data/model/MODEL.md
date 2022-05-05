# OCR模型转换
## 一、PaddleOCR的inference模型转onnx模型
### 环境依赖
#### 用户环境配置
     python == 3.7.5
	 静态图: paddlepaddle >= 1.8.0
	 动态图: paddlepaddle >= 2.0.0
	 onnx >= 1.7.0
#### 安装
##### 安装方式1
	 pip install paddle2onnx
##### 安装方式2
	 git clone https://github.com/PaddlePaddle/paddle2onnx.git
	 python setup.py install

#### 静态图模型导出
##### 下载预训练模型

本样例采用[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)的release/2.1分支作为基准，预训练模型请下载"PP-OCR 2.0 series model list（Update on Dec 15）"->"Chinese and English general OCR model (143.4M)" 对应的三个推理模型：
    Detection model: [DBNet](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar)
	Direction classifier model: [Mobilenet](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar)
	Recognition model: [CRNN](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar)

##### 检测模型转onnx模型

	paddle2onnx --model_dir ./ch_ppocr_server_v2.0_det_infer/ch_ppocr_server_v2.0_det_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_det_infer.onnx --opset_version 11 --enable_onnx_checker True

##### 方向分类模型转onnx模型

	paddle2onnx --model_dir ./ch_ppocr_server_v2.0_rec_infer/ch_ppocr_server_v2.0_rec_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_rec_infer.onnx --opset_version 11 --enable_onnx_checker True

##### 识别模型转onnx模型

	paddle2onnx --model_dir ./ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_mobile_v2.0_cls_infer.onnx --opset_version 11 --enable_onnx_checker True

## 二、onnx模型转om模型

### 检测模型
#### AIPP配置文件
	aipp_op {
	aipp_mode : static
	related_input_rank : 0
	input_format : RGB888_U8
	csc_switch : false
	rbuv_swap_switch : false

		mean_chn_0 :0
		mean_chn_1 :0
		mean_chn_2 :0

		min_chn_0 : 123.675
		min_chn_1 : 116.28
		min_chn_2 : 103.53

		var_reci_chn_0 : 0.017124753831664
		var_reci_chn_1 : 0.017507002801120
		var_reci_chn_2 : 0.017429193899782
	}
#### 转om模型命令
##### 转多分辨率档位模型
	/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./ch_ppocr_server_v2.0_det_infer.onnx --framework=5 --output_type=FP32 --output=Dynamic24_ch_ppocr_server_v2.0_det_infer --input_format=NCHW --input_shape="x:1,3,-1,-1" --dynamic_image_size="768,1024;1024,768;864,1120;1120,864;960,1216;1216,960;1056,1312;1312,1056;1152,1408;1408,1152;1248,1504;1504,1248;1344,1600;1600,1344;1440,1696;1696,1440;1536,1792;1792,1536;1632,1888;1888,1632;1728,1984;1984,1728;1824,2080;2080,1824" --soc_version=Ascend310 --insert_op_conf=./det_aipp.cfg
注意：可修改`--dynamic_image_size`后的档位列表，来适应数据集图片的尺寸范围，有利于改善精度；该paddle模型的尺寸宽高需设置为32的倍数

### 方向分类模型
#### AIPP配置文件
	aipp_op {
	aipp_mode : static
	related_input_rank : 0
	input_format : RGB888_U8
	csc_switch : false
	rbuv_swap_switch : false
	mean_chn_0 :0
	mean_chn_1 :0
	mean_chn_2 :0
	min_chn_0 :127.5
	min_chn_1 :127.5
	min_chn_2 :127.5
	var_reci_chn_0 : 0.00784313725490196
	var_reci_chn_1 : 0.00784313725490196
	var_reci_chn_2 : 0.00784313725490196
	}
#### 转om模型命令

##### 转动态batch模型
	/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./ch_ppocr_mobile_v2.0_cls_infer_3_48_192.onnx --framework=5 --output_type=FP32 --output=ch_ppocr_mobile_v2.0_cls_infer_3_48_192 --input_format=NCHW --input_shape="x:-1,3,48,192"  --dynamic_batch_size="1,2,4,8" --soc_version=Ascend310  --insert_op_conf="cls_aipp.cfg"

### 识别模型
#### AIPP配置文件
	aipp_op {
	aipp_mode : static
	related_input_rank : 0
	input_format : RGB888_U8
	csc_switch : false
	rbuv_swap_switch : false
	mean_chn_0 :0
	mean_chn_1 :0
	mean_chn_2 :0
	min_chn_0 :127.5
	min_chn_1 :127.5
	min_chn_2 :127.5
	var_reci_chn_0 : 0.00784313725490196
	var_reci_chn_1 : 0.00784313725490196
	var_reci_chn_2 : 0.00784313725490196
	}

#### 转om模型命令
##### 转多batch模型
	/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./ch_ppocr_server_v2.0_rec_infer.onnx --framework=5 --output_type=FP32 --output=ch_ppocr_server_v2.0_rec_infer_3_32_320_bs_1_2_4_8_16 --input_format=NCHW --input_shape="x:-1,3,32,320" --dynamic_batch_size="1,2,4,8,16" --soc_version=Ascend310 --insert_op_conf="rec_aipp.cfg"
注意：如果转换模型出现 `Prebuild op[LSTM_4/DynamicRnn] failed`，需要使用CANN 5.0.2及更高的版本
## 相关文档
- [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md "	Paddle2ONNX")
