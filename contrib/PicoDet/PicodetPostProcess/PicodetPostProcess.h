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
#ifndef PICODET_POST_PROCESS_H
#define PICODET_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace DefaultValues {
    const float DEFAULT_SCORE_THRESH = 0.4;
    const float DEFAULT_NMS_THRESH = 0.5;
    const uint32_t DEFAULT_STRIDES_NUM = 4;
    const uint32_t DEFAULT_CLASS_NUM = 80;
}

namespace MxBase {
    class PicodetPostProcess : public ObjectPostProcessBase
    {
    public:
        PicodetPostProcess() = default;

        ~PicodetPostProcess() = default;

        PicodetPostProcess(const PicodetPostProcess &other);

        PicodetPostProcess &operator=(const PicodetPostProcess &other);

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(const std::vector <MxBase::TensorBase> &tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {},
                          const std::map <std::string, std::shared_ptr<void>> &paramMap = {}) override;

    protected:
        bool IsValidTensors(const std::vector <MxBase::TensorBase> &tensors) const;

        APP_ERROR ObjectDetectionOutput(const std::vector <MxBase::TensorBase> &tensors,
                                   std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                                   const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {});

        void GetScoreAndLabel(const float *outBuffer, const uint32_t idx, float &score, int &curLabel);

        void GenerateBbox(const float *&bboxInfo, std::pair<int, int> center, int stride,
                          const ResizedImageInfo &resizedImageInfo,
                          ObjectInfo &objectInfo);

        APP_ERROR GetStrides(std::string &strStrides);

    protected:
        float scoreThresh_ = DefaultValues::DEFAULT_SCORE_THRESH;
        float nmsThresh_ = DefaultValues::DEFAULT_NMS_THRESH;
        uint32_t classNum_ = DefaultValues::DEFAULT_CLASS_NUM;
        uint32_t stridesNum_ = DefaultValues::DEFAULT_STRIDES_NUM;
        std::vector<float> strides_ = {};
    };
    extern "C" {
    std::shared_ptr<MxBase::PicodetPostProcess> GetObjectInstance();
    }
}
#endif