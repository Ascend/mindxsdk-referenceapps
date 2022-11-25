#!/bin/bash
# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mode='all'
epoch=100
style_dir="../datasets/style"       # 风格图片目录
content_dir="../datasets/content"   # 内容图片目录
code_dir="../datasets/code"         # 二维码图片目录
output_dir="./output"               # 输出目录
export OMP_NUM_THREADS=1
for code in ${code_dir}/*.jpg; do
    image_name=$(basename ${code} .jpg)
    code_img_path="${code_dir}/${image_name}.jpg"
    content_img_path="${content_dir}/${image_name}.jpg"
    if [ $mode == 'all' ]; then
        # 所有风格图测试
        for style in ${style_dir}/*.jpg; do
            style_name=$(basename ${style} .jpg)
            style_img_path="${style_dir}/${style_name}.jpg"
            output_img_path="${output_dir}/${image_name}_${style_name}"
            python -W ignore main.py --epoch ${epoch} --style_img_path ${style_img_path} --content_img_path ${content_img_path} --code_img_path ${code_img_path} --output_dir ${output_img_path}
        done
    elif [ $mode == 'some' ]; then
        # 部分风格图测试
        style_list=('candy.jpg' 'hb.jpg' 'been.jpg' 'st.jpg' 'square.jpg' 'udnie.jpg')
        for style in ${style_list[@]}; do
            style_name=$(basename ${style} .jpg)
            style_img_path="${style_dir}/${style_name}.jpg"
            output_img_path="${output_dir}/${image_name}_${style_name}"
            python -W ignore main.py --epoch ${epoch} --style_img_path ${style_img_path} --content_img_path ${content_img_path} --code_img_path ${code_img_path} --output_dir ${output_img_path}
        done
    else
        echo "Not supported test mode."
    fi
done
