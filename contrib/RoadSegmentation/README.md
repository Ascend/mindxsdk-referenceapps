# ·��ָ�
## 1. ���
����������MindX SDKʵ���˶˵��˵�·��ָ�ܣ���Ҫ������Unetģ�Ͷ������·��ͼƬ��������ָ���mask��Ĥ��Ȼ����ԭͼ��ϣ����ɱ�ע��·��Ŀ��ӻ������
## 2. Ŀ¼�ṹ
```
������ config #�����ļ�Ŀ¼
��   ������ aipp_road_segmentation.config
������ model  #ģ��Ŀ¼
�� 	������Road.onnx 
������ pipeline
��   ������ road.pipeline
������ plugin #������Ŀ¼
�� 	������RoadSegPostProcess
��		������ build
��		������ build.sh  #����ű�
��		������ lib 
��		��     ������plugins #����õĲ�����λ��
�� 		������ CMakeLists.txt
��   		������ MxpiRoadSegPostProcess.cpp
��  		������ MxpiRoadSegPostProcess.cpp.h
������ main.py
������ README.md
������ build.sh
������ run.sh
```
## 3. ����
| ������� | �汾   |
| :--------: | :------: |
|ubantu 18.04|18.04.1 LTS   |
|MindX SDK|2.0.4|
|Python|3.9.2|
|CANN|5.0.4|
|C++| 11.0|
|opencv2| |> 

## 4. ģ��ת��  
ʹ��Ascend atc���߽�onnxģ��ת��Ϊomģ��
```
# CANN��װĿ¼
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

# �� ����/modelĿ¼��ִ��atcת������ 
atc --framework=5 --model=Road.onnx --output=road_segmentation --input_format=NCHW  --insert_op_conf=../config/aipp_road_segmentation.config --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310  
```
������������Ϣ����ת���ɹ�
```
ATC run success
```
## 5. ����
### 5.1 ����MindX SDK��������
```
export MX_SDK_HOME=${SDK��װ·��}
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/driver/lib64/
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/MindX_SDK/mxVision/lib/plugins
export PYTHONPATH=${MX_SDK_HOME}/python:${PYTHONPATH}

#�鿴��������
env
```
### 5.2 ���������
```
��pluginĿ¼��
mkdir build
cd build
cmake ..
make -j
```
֮������� ����/plugin/lib/pluginsĿ¼�¿�������õ�.so���
```
#�����Ŀ¼
chmod 640 *.so #�޸�Ȩ��
```
Ȼ�����ƶ���${SDK��װ·��}/lib/pluginsĿ¼��

### 5.3 ����pipeline
�������賡��������pipeline�ļ�������·�������ȡ�
```
  #����mxpi_tensorinfer�����ģ�ͼ���·���� modelPath
  "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "${road_segmentation.omģ��·��}"
            },
            "factory": "mxpi_tensorinfer",
            "next": "appsink0"
        },
	#���ÿ��ӻ�������·����location
	"appsink0": {
            "props": {
                "blocksize": "4096000",
	"location":"${�������ļ���}" 
            },
            "factory": "filesink"
        }
```
### 5.4 ����ͼƬ
���޸�main.py�ļ��е�filepath�ı���ֵ�����Ϊ����ͼƬ�ĵ�ַ
### 5.5 ������
```
#������Ŀ¼��ִ��
python3.9 main.py
```
�ɿ���·��ָ���ӻ���� 




