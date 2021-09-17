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

TP = 0
FP = 0
FN = 0
with open("result.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        name = line.split(' ')[0]
        label = int(line.split(' ')[1])
        pred = int(line.split(' ')[-1])
        if label == 1:
            if pred == 1:
                TP += 1
            else:
                FN += 1
        elif label == 0:
            if pred == 1:
                FP += 1
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print('precision:', precision)
print('recall:', recall)
