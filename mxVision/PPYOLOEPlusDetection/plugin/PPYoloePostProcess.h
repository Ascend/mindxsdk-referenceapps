/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef SAMPLE_POST_PROCESS_H
#define SAMPLE_POST_PROCESS_H

#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace MxBase {
const float DEFAULT_OBJECTNESS_THRESH = 0.3;
const float DEFAULT_IOU_THRESH = 0.45;

class PPYoloePostProcess : public ObjectPostProcessBase {
public:
    PPYoloePostProcess();

    ~PPYoloePostProcess() {}

    PPYoloePostProcess(const PPYoloePostProcess &other);

    PPYoloePostProcess &operator=(const PPYoloePostProcess &other);

    SDK_DEPRECATED_FOR() APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;
    APP_ERROR Init(const std::map<std::string, std::string> &postConfig) override;

    APP_ERROR DeInit() override;

    APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<ObjectInfo>> &objectInfos,
                      const std::vector<ResizedImageInfo> &resizedImageInfos = {},
                      const std::map<std::string, std::shared_ptr<void>> &paramMap = {}) override;

private:
    void LogObjectInfo(std::vector<std::vector<ObjectInfo>> &objectInfos);
    void ConstructBoxFromOutput(float *output, float *boxOutput, size_t offset,
                                std::vector<ObjectInfo> &objectInfo, const ResizedImageInfo &resizedImageInfo);
    float objectnessThresh_ = DEFAULT_OBJECTNESS_THRESH;
    float iouThresh_ = DEFAULT_IOU_THRESH;
};

extern "C" {
std::shared_ptr<MxBase::PPYoloePostProcess> GetObjectInstance();
}
}
#endif