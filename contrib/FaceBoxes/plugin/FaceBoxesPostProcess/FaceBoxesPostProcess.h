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

#ifndef FaceBoxes_POST_PROCESS_H
#define FaceBoxes_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "opencv2/opencv.hpp"
namespace
{
    const int DEFAULT_OBJECT_CONF_TENSOR = 1;
    const int DEFAULT_OBJECT_INFO_TENSOR = 0;
    const float DEFAULT_IOU_THRESH = 0.3;
    const float DEFAULT_CONFIDENCE_THRESH = 0.5;
}


namespace MxBase
{   
    bool operator<(const ObjectInfo &a ,const ObjectInfo &b){
        return a.confidence < b.confidence;
    }
        
    class FaceboxesPostProcess : public ObjectPostProcessBase
    {
    public:
        FaceboxesPostProcess() = default;

        ~FaceboxesPostProcess() = default;

        FaceboxesPostProcess(const FaceboxesPostProcess& other) = default;

        FaceboxesPostProcess& operator=(const FaceboxesPostProcess& other);

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>>& postConfig) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<ObjectInfo>> &objectInfos,
                      const std::vector<ResizedImageInfo> &resizedImageInfos = {},
                      const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;

    protected:
        void GeneratePriorBox(cv::Mat &anchors);
        cv::Mat decode(cv::Mat &loc, cv::Mat &prior, float resize_scale_factor);
        void ObjectDetectionOutput(const std::vector <MxBase::TensorBase>& tensors,
            std::vector <std::vector<MxBase::ObjectInfo>>& objectInfos,
            const std::vector <MxBase::ResizedImageInfo>& resizedImageInfos = {});
    private:
        uint32_t objectConfTensor_ = DEFAULT_OBJECT_CONF_TENSOR;
        uint32_t objectInfoTensor_ = DEFAULT_OBJECT_INFO_TENSOR;
        float iouThresh_ = DEFAULT_IOU_THRESH;
        float confThresh_ = DEFAULT_CONFIDENCE_THRESH;
    };

    extern "C" {
        std::shared_ptr<MxBase::FaceboxesPostProcess> GetObjectInstance();
    }

}
#endif
