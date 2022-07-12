/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#ifndef KPYunet_POST_PROCESS_H
#define KPYunet_POST_PROCESS_H
#include "MxBase/PostProcessBases/KeypointPostProcessBase.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "opencv2/opencv.hpp"
#define DEFAULT_OBJECT_CONF_TENSOR  1
#define DEFAULT_OBJECT_INFO_TENSOR  0
#define DEFAULT_IOU_THRESH  0.3
#define DEFAULT_CONFIDENCE_THRESH  0.9

namespace MxBase {
    bool operator<(const KeyPointDetectionInfo &a, const KeyPointDetectionInfo &b) {
        return a.score < b.score;
    }
    bool operator<(const ObjectInfo &a, const ObjectInfo &b) {
        return a.confidence < b.confidence;
    }
    
    class KPYunetPostProcess : public KeypointPostProcessBase {
    public:
        KPYunetPostProcess() = default;

        ~KPYunetPostProcess() = default;

        KPYunetPostProcess(const KPYunetPostProcess& other);

        KPYunetPostProcess& operator=(const KPYunetPostProcess& other);

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>>& postConfig) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<KeyPointDetectionInfo>> &keypointInfos,
                          const std::vector<ResizedImageInfo> &resizedImageInfos = {},
                          const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;

    protected:
        void GeneratePriorBox(cv::Mat &anchors);
        cv::Mat decode_for_loc(cv::Mat &loc, cv::Mat &prior);
        void KeypointDetectionOutput(const std::vector <MxBase::TensorBase>& tensors,
                                     std::vector <std::vector<MxBase::KeyPointDetectionInfo>>& ketpointInfos,
                                     const std::vector <MxBase::ResizedImageInfo>& resizedImageInfos = {});
        void generate_keypointInfos(const std::vector <TensorBase>& tensors,
                                    std::vector <std::vector<KeyPointDetectionInfo>>& keypointInfos,
                                    const std::vector <ResizedImageInfo>& resizedImageInfos,
                                    cv::Mat &res);
    private:
        uint32_t objectConfTensor_ = DEFAULT_OBJECT_CONF_TENSOR;
        uint32_t objectInfoTensor_ = DEFAULT_OBJECT_INFO_TENSOR;
        float iouThresh_ = DEFAULT_IOU_THRESH;
        float confThresh_ = DEFAULT_CONFIDENCE_THRESH;
    };

    extern "C" {
        std::shared_ptr<MxBase::KPYunetPostProcess> GetKetpointInstance();
    }
}
#endif
