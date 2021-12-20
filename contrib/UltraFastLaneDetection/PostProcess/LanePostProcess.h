 /*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LANE_POST_PROCESS_H
#define LANE_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"


namespace MxBase {
    class LanePostProcess : public ObjectPostProcessBase
    {
    public:
        LanePostProcess() = default;

        ~LanePostProcess() = default;

        LanePostProcess(const LanePostProcess &other) = default;

        LanePostProcess &operator=(const LanePostProcess &other) = default;

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(const std::vector <MxBase::TensorBase> &tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {},
                          const std::map <std::string, std::shared_ptr<void>> &paramMap = {}) override;

    protected:
        void my_softmax(int x,int y);

        void ObjectDetectionOutput(const std::vector <MxBase::TensorBase> &tensors,
                                   std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                                   const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {});

    };
    extern "C" {
    std::shared_ptr<MxBase::LanePostProcess> GetObjectInstance();
    }
}
#endif