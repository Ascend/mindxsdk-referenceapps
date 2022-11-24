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

#!/bin/bash
MODE=$1
echo ${MODE}
if [ ! ${MODE} ]; then
    MODE="./mvtec"
    echo ${MODE}
else
    echo ${MODE} 
fi

echo "1/15 test bottle"
python eval_precision.py -d bottle --dataset_path ${MODE}

echo "2/15 test cable"
python eval_precision.py -d cable --dataset_path ${MODE}

echo "3/15 test capsule"
python eval_precision.py -d capsule --dataset_path ${MODE}

echo "4/15 test carpet"
python eval_precision.py -d carpet --dataset_path ${MODE}

echo "5/15 test grid"
python eval_precision.py -d grid --dataset_path ${MODE}

echo "6/15 test hazelnut"
python eval_precision.py -d hazelnut --dataset_path ${MODE}

echo "7/15 test leather"
python eval_precision.py -d leather --dataset_path ${MODE}

echo "8/15 test metal_nut"
python eval_precision.py -d metal_nut --dataset_path ${MODE}

echo "9/15 test pill"
python eval_precision.py -d pill --dataset_path ${MODE}

echo "10/15 test screw"
python eval_precision.py -d screw --dataset_path ${MODE}

echo "11/15 test tile"
python eval_precision.py -d tile --dataset_path ${MODE}

echo "12/15 test toothbrush"
python eval_precision.py -d toothbrush --dataset_path ${MODE}

echo "13/15 test transistor"
python eval_precision.py -d transistor --dataset_path ${MODE}

echo "14/15 test wood"
python eval_precision.py -d wood --dataset_path ${MODE}

echo "15/15 test zipper"
python eval_precision.py -d zipper --dataset_path ${MODE}

python calculate_txt_avg.py -m precision