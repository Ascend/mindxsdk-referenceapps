#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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


echo "1/15 test bottle"
python eval_performance.py -d bottle

echo "2/15 test cable"
python eval_performance.py -d cable

echo "3/15 test capsule"
python eval_performance.py -d capsule

echo "4/15 test carpet"
python eval_performance.py -d carpet

echo "5/15 test grid"
python eval_performance.py -d grid

echo "6/15 test hazelnut"
python eval_performance.py -d hazelnut

echo "7/15 test leather"
python eval_performance.py -d leather

echo "8/15 test metal_nut"
python eval_performance.py -d metal_nut

echo "9/15 test pill"
python eval_performance.py -d pill

echo "10/15 test screw"
python eval_performance.py -d screw

echo "11/15 test tile"
python eval_performance.py -d tile

echo "12/15 test toothbrush"
python eval_performance.py -d toothbrush

echo "13/15 test transistor"
python eval_performance.py -d transistor

echo "14/15 test wood"
python eval_performance.py -d wood

echo "15/15 test zipper"
python eval_performance.py -d zipper

python calculate_txt_avg.py -m performance
