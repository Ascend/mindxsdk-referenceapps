# Text Detection Model(CTPN)

## pb模型链接

```
https://github.com/MaybeShewill-CV/CRNN_Tensorflow

参考: https://www.hiascend.com/zh/software/modelzoo/models/detail/1/2838f191714846258fb54f51d1fe0ca5/1
```



## pb模型转om模型

#### AIPP配置文件

	aipp_yuv.cfg:
		aipp_op {
			aipp_mode : static
			related_input_rank : 0
			input_format : YUV420SP_U8
			crop : false
			csc_switch : true
			rbuv_swap_switch : false
			matrix_r0c0 : 256
			matrix_r0c1 : 0
			matrix_r0c2 : 359
			matrix_r1c0 : 256
			matrix_r1c1 : -88
			matrix_r1c2 : -183
			matrix_r2c0 : 256
			matrix_r2c1 : 454
			matrix_r2c2 : 0
			input_bias_0 : 0
			input_bias_1 : 128
			input_bias_2 : 128
		}

#### 转om模型命令

```shell
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=ctpn_tf.pb \
	--output=ctpn_yuv \
	--framework=3 \
	--input_shape="input_image:1,608,1072,3" \
	--insert_op_conf=./aipp_yuv.cfg \
	--soc_version=Ascend310 \
	--log=error \
	--enable_scope_fusion_passes=ScopeDynamicRNNPass
```

# Text Recognition Model（CRNN）

## PaddleOCR的inference模型转onnx模型

### 环境依赖

#### 用户环境配置

	 python == 3.9.2
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

##### 检测模型转onnx模型

	paddle2onnx --model_dir ./ch_ppocr_server_v2.0_det_infer/ch_ppocr_server_v2.0_det_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_det_infer.onnx --opset_version 11 --enable_onnx_checker True

##### 方向分类模型转onnx模型

	paddle2onnx --model_dir ./ch_ppocr_server_v2.0_rec_infer/ch_ppocr_server_v2.0_rec_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_server_v2.0_rec_infer.onnx --opset_version 11 --enable_onnx_checker True

##### 识别模型转onnx模型

	paddle2onnx --model_dir ./ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./ch_ppocr_mobile_v2.0_cls_infer.onnx --opset_version 11 --enable_onnx_checker True

## onnx模型转om模型

### 检测模型

#### AIPP配置文件

	aipp_op {
		aipp_mode: static
		input_format : YUV420SP_U8
		src_image_size_w : 1312
		src_image_size_h : 736
		crop: false
		csc_switch : true
		rbuv_swap_switch : false
		matrix_r0c0 : 256
		matrix_r0c1 : 454
		matrix_r0c2 : 0
		matrix_r1c0 : 256
		matrix_r1c1 : -88
		matrix_r1c2 : -183
		matrix_r2c0 : 256
		matrix_r2c1 : 0
		matrix_r2c2 : 359
		input_bias_0 : 0
		input_bias_1 : 128
		input_bias_2 : 128
	
		mean_chn_0 :0
		mean_chn_1 :0
		mean_chn_2 :0
	
		min_chn_0 : 103.53
		min_chn_1 : 116.28
		min_chn_2 : 123.675
	
		var_reci_chn_0 : 0.017429193899782
		var_reci_chn_1 : 0.017507002801120
		var_reci_chn_2 : 0.017124753831664
	}

#### 转om模型命令

	/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./ch_ppocr_server_v2.0_det_infer.onnx --framework=5 --output_type=FP32 --output=dvaipp_ch_ppocr_server_v2.0_det_infer --input_format=NCHW --input_shape="x:1,3,736,1312" --soc_version=Ascend310 --log=debug --insert_op_conf=./db_detection_dvpp.aippconfig

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

##### 转单batch模型

	/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=../onnx_models/ch_ppocr_mobile_v2.0_cls_infer_3_48_192.onnx --framework=5 --output_type=FP32 --output=ch_ppocr_mobile_v2.0_cls_infer_bs1_3_48_192 --input_format=NCHW --input_shape="x:1,3,48,192" --soc_version=Ascend310 --insert_op_conf="cls_aipp.cfg"


##### 转动态batch模型

	/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=../onnx_models/ch_ppocr_mobile_v2.0_cls_infer_3_48_192.onnx --framework=5 --output_type=FP32 --output=ch_ppocr_mobile_v2.0_cls_infer_3_48_192 --input_format=NCHW --input_shape="x:-1,3,48,192"  --dynamic_batch_size="1,2,4,8" --soc_version=Ascend310  --insert_op_conf="cls_aipp.cfg"

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

##### 转单batch模型

	atc --model=./ch_ppocr_server_v2.0_rec_infer_modify.onnx --framework=5 --output_type=FP32 --output=ch_ppocr_server_v2.0_rec_infer_modify_bs1 --input_format=NCHW --input_shape="x:1,3,32,320" --soc_version=Ascend310 --insert_op_conf="rec_aipp.cfg"

##### 转动态batch模型

	atc --model=./ch_ppocr_server_v2.0_rec_infer_modify.onnx --framework=5 --output_type=FP32 --output=ch_ppocr_server_v2.0_rec_infer_modify_dy_bs --input_format=NCHW --input_shape="x:-1,3,32,320" --dynamic_batch_size="1,2,4,8" --soc_version=Ascend310 --insert_op_conf="rec_aipp.cfg"

# 语言模型（Bert）

## Bert模型链接

```
https://github.com/google-research/bert
```

## ckpt模型转pb模型

```python
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
 
def freeze_graph(ckpt, output_graph):
    output_node_names = 'bert/encoder/Reshape_13'
    # saver = tf.train.import_meta_graph(ckpt+'.meta', clear_devices=True)
    saver = tf.compat.v1.train.import_meta_graph(ckpt+'.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
 
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(',')
        )
        with tf.gfile.GFile(output_graph, 'wb') as fw:
            fw.write(output_graph_def.SerializeToString())
        print ('{} ops in the final graph.'.format(len(output_graph_def.node)))
 
ckpt = './bert_model.ckpt'
pb   = './bert_model_new.pb'
 
if __name__ == '__main__':
    freeze_graph(ckpt, pb)
```

## 转om模型命令

```shell
atc --model=./bert_model_new.pb \
--framework=3 --input_shape "Placeholder:1,128;Placeholder_1:1,128;Placeholder_2:1,128" \
--output=model_bert --soc_version=Ascend310
```



## 相关文档

- [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md "	Paddle2ONNX")

