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

epoch=100
style_img_path="../datasets/style/udnie.jpg"
content_img_path="../datasets/content/panda.jpg"
code_img_path="../datasets/code/panda.jpg"
output_dir="./output/panda/"
export OMP_NUM_THREADS=1
python -W ignore main.py --epoch ${epoch} --style_img_path ${style_img_path} --content_img_path ${content_img_path} --code_img_path ${code_img_path} --output_dir ${output_dir}