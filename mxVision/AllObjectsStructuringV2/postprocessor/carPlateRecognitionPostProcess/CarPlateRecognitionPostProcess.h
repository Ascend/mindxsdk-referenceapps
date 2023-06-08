/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CARPLATERECOGNITIONPOSTPROCESS_CARPLATERECOGNITIONPOSTPROCESS_H
#define CARPLATERECOGNITIONPOSTPROCESS_CARPLATERECOGNITIONPOSTPROCESS_H
#include "MxBase/MxBase.h"

const int CAR_PLATE_CHARS_NUM = 65;
const std::string CAR_PLATE_CHARS[CAR_PLATE_CHARS_NUM] = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
    "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
    "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1",
    "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D",
    "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
};

class CarPlateRecognitionPostProcess {
public:
    CarPlateRecognitionPostProcess();

    ~CarPlateRecognitionPostProcess() = default;

    APP_ERROR Init();

    APP_ERROR Process(const std::vector<MxBase::Tensor> &inferOutputs, std::string& carPlateRes);
};

#endif //CARPLATERECOGNITIONPOSTPROCESS_CARPLATERECOGNITIONPOSTPROCESS_H