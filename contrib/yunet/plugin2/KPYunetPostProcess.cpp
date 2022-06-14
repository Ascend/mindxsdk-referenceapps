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

#include "KPYunetPostProcess.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t LEFTTOPX = 0;
    const uint32_t LEFTTOPY = 1;
    const uint32_t RIGHTTOPX = 2;
    const uint32_t RIGHTTOPY = 3;
    const int PRIOR_PARAMETERS[4][3] = {{10, 16, 24}, {32, 48, -1}, {64, 96, -1}, {128, 192, 256}};
    const int PRIOR_PARAMETERS_COUNT = 3;
    const float IMAGE_WIDTH = 160.0;
    const float IMAGE_HEIGHT = 120.0;
    const float STEPS[4] = {8.0, 16.0, 32.0, 64.0};
    const float VARIANCE[2] = {0.1, 0.2};
    const uint32_t RECTANGLEPOINT = 4;
    const uint32_t KEYPOINTNUM = 5;
    const uint32_t DIM = 2;
    const uint32_t RECTANGLE_COLOR = 1;
    const uint32_t KEYPOINT_COLOR = 2;
    const uint32_t DIV_TWO = 2;
}
namespace MxBase {
    KPYunetPostProcess& KPYunetPostProcess::operator=(const KPYunetPostProcess& other)
    {
        if (this == &other) {
            return *this;
        }
        KeypointPostProcessBase::operator=(other);
        return *this;
    }

    APP_ERROR KPYunetPostProcess::Init(const std::map <std::string, std::shared_ptr<void>>& postConfig)
    {
        LogInfo << "Start to Init KPYunetPostProcess.";
        APP_ERROR ret = KeypointPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in KeypointPostProcessBase.";
            return ret;
        }
        LogInfo << "End to Init KPYunetPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR KPYunetPostProcess::DeInit()
    {
        return APP_ERR_OK;
    }

    void KPYunetPostProcess::generate_keypointInfos(const std::vector <TensorBase>& tensors,
                                                    std::vector <std::vector<KeyPointDetectionInfo>>& keypointInfos,
                                                    const std::vector <ResizedImageInfo>& resizedImageInfos,
                                                    cv::Mat &res)
    {
        auto shape = tensors[0].GetShape();
        float width_resize_scale = (float)resizedImageInfos[0].widthResize / resizedImageInfos[0].widthOriginal;
        float height_resize_scale = (float)resizedImageInfos[0].heightResize / resizedImageInfos[0].heightOriginal;
        uint32_t batchSize = shape[0];
        uint32_t VectorNum = shape[1];

        for (uint32_t i = 0; i < batchSize; i++) {
            std::vector <ObjectInfo> objectInfo, objectInfoSorted;
            std::vector <KeyPointDetectionInfo> keypointInfoSorted;
            auto dataPtr_Conf = (float *) tensors[1].GetBuffer() + i * tensors[1].GetByteSize() / batchSize;
            for (uint32_t j = 0; j < VectorNum; j++) {
                float* begin_Conf = dataPtr_Conf + j * 2;
                float conf = *(begin_Conf + 1);
<<<<<<< HEAD
=======

>>>>>>> 544442555aec11d25c39000a6ff5bd90fd5e1a0b
                if (conf> confThresh_) {
                    ObjectInfo objInfo;
                    objInfo.confidence = conf;
                    objInfo.x0 = res.at<float>(j, LEFTTOPX) * IMAGE_WIDTH / width_resize_scale;
                    objInfo.y0 = res.at<float>(j, LEFTTOPY) * IMAGE_HEIGHT / height_resize_scale;
                    objInfo.x1 = res.at<float>(j, RIGHTTOPX) * IMAGE_WIDTH / width_resize_scale;
                    objInfo.y1 = res.at<float>(j, RIGHTTOPY) * IMAGE_HEIGHT / height_resize_scale;
                    objInfo.classId = j;
                    
                    objectInfo.push_back(objInfo);
                }
            }

            MxBase::NmsSort(objectInfo, iouThresh_);
            for (uint32_t j = 0; j < objectInfo.size(); j++) {
                int keypoint_Pos = objectInfo[j].classId;
                objectInfo[j].classId = RECTANGLE_COLOR;
                objectInfoSorted.push_back(objectInfo[j]);

                KeyPointDetectionInfo kpInfo;
                float* begin_Conf = dataPtr_Conf + keypoint_Pos * 2;
                kpInfo.score = *(begin_Conf + 1);

                for (int k = 0; k < KEYPOINTNUM; k++)
                {
                    float x = res.at<float>(keypoint_Pos, RECTANGLEPOINT + k * DIM) * IMAGE_WIDTH / width_resize_scale;
                    float y = res.at<float>(keypoint_Pos, RECTANGLEPOINT + k * DIM + 1) * IMAGE_HEIGHT / height_resize_scale;
                    kpInfo.keyPointMap[k].push_back(x);
                    kpInfo.keyPointMap[k].push_back(y);
                }
                keypointInfoSorted.push_back(kpInfo);
            }
            keypointInfos.push_back(keypointInfoSorted);
        }
    }

    void KPYunetPostProcess::KeypointDetectionOutput(const std::vector <TensorBase>& tensors,
                                                     std::vector <std::vector<KeyPointDetectionInfo>>& keypointInfos,
                                                     const std::vector <ResizedImageInfo>& resizedImageInfos)
    {
        LogInfo << "KPYunetPostProcess start to write results.";
        
        auto loc = tensors[0].GetBuffer();
        auto conf = tensors[1].GetBuffer();
        auto shape = tensors[0].GetShape();
        auto iou = tensors[2].GetBuffer();

        cv::Mat PriorBox;
        cv::Mat location = cv::Mat(shape[1], shape[2], CV_32FC1, tensors[0].GetBuffer());
        GeneratePriorBox(PriorBox);
        cv::Mat res = decode_for_loc(location, PriorBox);
        generate_keypointInfos(tensors, keypointInfos, resizedImageInfos, res);

        LogInfo << "KPYunetPostProcess write results successed.";
    }
    APP_ERROR KPYunetPostProcess::Process(const std::vector<TensorBase> &tensors,
                                          std::vector<std::vector<KeyPointDetectionInfo>> &KeypointInfos,
                                          const std::vector<ResizedImageInfo> &resizedImageInfos,
                                          const std::map<std::string, std::shared_ptr<void>> &configParamMap)
    {
        LogInfo << "Start to Process KPYunetPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << "CheckAndMoveTensors failed. ret=" << ret;
            return ret;
        }
        KeypointDetectionOutput(inputs, KeypointInfos, resizedImageInfos);
        LogInfo << "End to Process KPYunetPostProcess.";
        return APP_ERR_OK;
    }

    /*
     * @description: Generate prior boxes for detection boxes decoding
     * @param anchors  A Matrix used to save prior boxes that contains box coordinates(x0,y0,x1,y1), shape[21824,4]
     */
    void KPYunetPostProcess::GeneratePriorBox(cv::Mat &anchors)
    {
        std::vector<std::vector<int>>feature_maps(RECTANGLEPOINT, std::vector<int>(DIM));
        for (int i = 0; i < feature_maps.size(); i++) {
            feature_maps[i][0] = IMAGE_HEIGHT / STEPS[i];
            feature_maps[i][1] = IMAGE_WIDTH / STEPS[i];
        }
        for (int k = 0; k < feature_maps.size(); k++) {
            auto f = feature_maps[k];
            float step = (float)STEPS[k];
            for (int i = 0; i < f[0]; i++) {
                for (int j = 0; j < f[1]; j++) {
                    for (int l = 0; l < PRIOR_PARAMETERS_COUNT && PRIOR_PARAMETERS[k][l] != -1; l++) {
                        float min_size = PRIOR_PARAMETERS[k][l];
                        cv::Mat anchor(1, RECTANGLEPOINT, CV_32F);
                        float center_x = (j + 0.5f) * step;
                        float center_y = (i + 0.5f) * step;

                        float xmin = (center_x - min_size / 2.f) / IMAGE_WIDTH;
                        float ymin = (center_y - min_size / 2.f) / IMAGE_HEIGHT;
                        float xmax = (center_x + min_size / 2.f) / IMAGE_WIDTH;
                        float ymax = (center_y + min_size / 2.f) / IMAGE_HEIGHT;

                        float prior_width = xmax - xmin;
                        float prior_height = ymax - ymin;
                        float prior_center_x = (xmin + xmax) / 2;
                        float prior_center_y = (ymin + ymax) / 2;

                        anchor.at<float>(0, LEFTTOPX) = prior_width;
                        anchor.at<float>(0, LEFTTOPY) = prior_height;
                        anchor.at<float>(0, RIGHTTOPX) = prior_center_x;
                        anchor.at<float>(0, RIGHTTOPY) = prior_center_y;

                        anchors.push_back(anchor);
                    }
                }
            }
        }
    }

    cv::Mat KPYunetPostProcess::decode_for_loc(cv::Mat &loc, cv::Mat &prior) {
        cv::Mat loc_first = loc.colRange(0, 2);
        cv::Mat loc_last = loc.colRange(2, 4);
        cv::Mat prior_first = prior.colRange(0, 2);
        cv::Mat prior_last = prior.colRange(2, 4);
        cv::Mat facepoint = loc.colRange(4, 14);

        cv::Mat boxes1 = prior_last + (loc_first * VARIANCE[0]).mul(prior_first);
        cv::Mat boxes2;
        cv::exp(loc_last * VARIANCE[1], boxes2);
        boxes2 = boxes2.mul(prior_first);
        boxes1 = boxes1 - boxes2 / DIV_TWO;
        boxes2 = boxes2 + boxes1;

        cv::Mat boxes3;
        for (int i = 0; i < KEYPOINTNUM; i++)
        {
            cv::Mat singlepoint = facepoint.colRange(i * 2, (i + 1) * 2);
            singlepoint = prior_last + (singlepoint * VARIANCE[0]).mul(prior_first);
            if (i == 0) boxes3 = singlepoint;
            else cv::hconcat(boxes3, singlepoint, boxes3);
        }

        cv::Mat boxes;
        cv::hconcat(boxes1, boxes2, boxes);
        cv::hconcat(boxes, boxes3, boxes);
        return boxes;
    }

    extern "C" {
        std::shared_ptr <MxBase::KPYunetPostProcess> GetKeypointInstance()
        {
            LogInfo << "Begin to get KPYunetPostProcess instance.";
            auto instance = std::make_shared<MxBase::KPYunetPostProcess>();
            LogInfo << "End to get KPYunetPostProcess instance.";
            return instance;
        }
    }
}
