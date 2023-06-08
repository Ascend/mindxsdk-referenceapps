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

#ifndef FACELANDMARK_FACELANDMARKPOSTPROCESS_H
#define FACELANDMARK_FACELANDMARKPOSTPROCESS_H
#include "MxBase/MxBase.h"

struct KeyPointAndAngle{
    std::vector<float> keyPoints;
    float angleYaw = 0.0;
    float anglePitch = 0.0;
    float angleRoll = 0.0;
};

class FaceLandmarkPostProcess {
public:
    FaceLandmarkPostProcess();

    ~FaceLandmarkPostProcess() {}

    APP_ERROR Init();

    APP_ERROR DeInit();

    APP_ERROR Process(std::vector<MxBase::Tensor>& inferOutputs,
                      KeyPointAndAngle& keyPointAndAngle); 
};

#endif //FACELANDMARK_FACELANDMARKPOSTPROCESS_H

