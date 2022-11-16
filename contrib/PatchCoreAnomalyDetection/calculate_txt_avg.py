# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

import argparse
parser = argparse.ArgumentParser(description='train')

parser.add_argument('--mode', "-m", type=str, default="precision")
args = parser.parse_args()


def calculate_precision(path):
    cate_cnt = 0
    auroc_avg_max = 0
    auroc_avg_topk_10 = 0
    for line in open(path, 'r'):
        cate_cnt += 1
        auroc_avg_max += float(line.split("\t")[2])
        auroc_avg_topk_10 += float(line.split("\t")[4])
    auroc_avg_topk_10 = auroc_avg_topk_10 / cate_cnt
    auroc_avg_max = auroc_avg_max / cate_cnt
    print(f"auroc:{auroc_avg_max}\tauroc_topK10:{auroc_avg_topk_10}")


def calculate_performance(path):
    cate_cnt = 0
    avg_time = 0
    for line in open(path, 'r'):
        cate_cnt += 1
        avg_time += float(line.split("\t")[2])
    avg_time = avg_time / cate_cnt
    fps = 1 / avg_time
    print(f"avg_time:{avg_time}\tfps:{fps}")


if __name__ == '__main__':
    if args.mode == "precision":
        calculate_precision("precision.txt")
    elif args.mode == "performance":
        calculate_performance("performance.txt")
    else:
        print("please check the mode!")
