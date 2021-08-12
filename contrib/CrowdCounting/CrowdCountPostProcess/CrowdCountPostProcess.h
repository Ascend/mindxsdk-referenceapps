/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef CrowdCountPicture_POST_PROCESS_H
#define CrowdCountPicture_POST_PROCESS_H
#include "MxBase/PostProcessBases/PostProcessBase.h"
#include <opencv2/opencv.hpp>

namespace {
const int MODEL_WIDTH = 1408;
const int MODEL_HEIGHT = 800;
}
struct NetInfo{
    int netWidth;
    int netHeight;
    int classNum;
};
class CrowdCountPostProcess : public MxBase::PostProcessBase {
public:
    CrowdCountPostProcess() = default;
    ~CrowdCountPostProcess() = default;
    CrowdCountPostProcess(const CrowdCountPostProcess &other) = default;
    CrowdCountPostProcess &operator=(const CrowdCountPostProcess &other);
    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;
    APP_ERROR DeInit() override;
    APP_ERROR Process(const std::vector<MxBase::TensorBase> &tensors,std::vector<MxBase::TensorBase> &outputs,
		      std::vector<int> &results);

protected:
    int modelWidth_ = MODEL_WIDTH;
    int modelHeight_ = MODEL_HEIGHT;
};
#endif



