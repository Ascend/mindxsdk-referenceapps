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

#ifndef Solov2_POST_PROCESS_H
#define Solov2_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include <math.h>

namespace DefaultValues {
    const int DEFAULT_CLASS_NUM = 80;
    const int DEFAULT_HEIGHT = 800;
    const int DEFAULT_WIDTH = 1216;
    const float DEFAULT_SCORE_THRESH = 0.3;
}

namespace MxBase {
    class Solov2PostProcess: public ObjectPostProcessBase {

    public:
        Solov2PostProcess() = default;

        ~Solov2PostProcess() = default;

        Solov2PostProcess(const Solov2PostProcess &other);

        Solov2PostProcess &operator=(const Solov2PostProcess &other);

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>> &postConfig);

        APP_ERROR DeInit();

        APP_ERROR Process(const std::vector <MxBase::TensorBase> &tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {},
                          const std::map <std::string, std::shared_ptr<void>> &paramMap = {});

    protected:
        bool IsValidTensors(const std::vector <MxBase::TensorBase> &tensors) const;
        void ReadDataFromTensor(const std::vector <MxBase::TensorBase> &tensors,
                                std::vector<std::vector<std::vector<uint8_t>>> &seg,
                                std::vector<int> &label, std::vector<float> &score);

        void ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                   std::vector<std::vector<ObjectInfo>> &objectInfos,
                                   const std::vector<ResizedImageInfo> &resizedImageInfos);
        void GenerateBoxes(std::vector<std::vector<std::vector<uint8_t>>> &seg,
                           std::vector<int> &label, std::vector<float> &score,
                           std::vector <MxBase::ObjectInfo> &detBoxes);


    protected:
        int ori_h_{0};
        int ori_w_{0};
        int img_h_{0};
        int img_w_{0};
        int height_ = DefaultValues::DEFAULT_HEIGHT;
        int width_ = DefaultValues::DEFAULT_WIDTH;
        int classNum_ = DefaultValues::DEFAULT_CLASS_NUM;
        float scoreThresh_ = DefaultValues::DEFAULT_SCORE_THRESH; // Confidence threhold
    };
    extern "C" {
    std::shared_ptr<MxBase::Solov2PostProcess> GetObjectInstance();
    }
}
#endif