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

#include "YunetPostProcess.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t LEFTTOPX = 0;
    const uint32_t LEFTTOPY = 1;
    const uint32_t RIGHTTOPX = 2;
    const uint32_t RIGHTTOPY = 3;
    const int PRIOR_PARAMETERS[4][3] = {{10, 16, 24}, {32, 48, -1}, {64, 96, -1}, {128, 192, 256}}; 
    const float IMAGE_WIDTH = 1920.0;
    const float IMAGE_HEIGHT = 1080.0;
    const float STEPS[4] = {8.0, 16.0, 32.0, 64.0};
    const float VARIANCE[2] = {0.1, 0.2};
    const uint32_t RECTANGLEPOINT = 4;
    const uint32_t DIM = 2; // image dimension is 2
    const uint32_t RECTANGLE_COLOR = 1;
    const uint32_t KEYPOINT_COLOR = 2;
    const uint32_t DIV_TWO = 2;
}
namespace MxBase {
    YunetPostProcess& YunetPostProcess::operator=(const YunetPostProcess& other)
    {
        if (this == &other) {
            return *this;
        }
        ObjectPostProcessBase::operator=(other);
        return *this;
    }

    APP_ERROR YunetPostProcess::Init(const std::map <std::string, std::shared_ptr<void>>& postConfig)
    {
        LogInfo << "Start to Init YunetPostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        LogInfo << "End to Init YunetPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR YunetPostProcess::DeInit()
    {
        return APP_ERR_OK;
    }

    void YunetPostProcess::ObjectDetectionOutput(const std::vector <TensorBase>& tensors,
                                                 std::vector <std::vector<ObjectInfo>>& objectInfos,
                                                 const std::vector <ResizedImageInfo>& resizedImageInfos)
    {
        LogInfo << "YunetPostProcess start to write results.";
                
        for (auto num : { objectInfoTensor_, objectConfTensor_ }) {
            if ((num >= tensors.size()) || (num < 0)) {
                LogError << GetError(APP_ERR_INVALID_PARAM) << "TENSOR(" << num
                    << ") must ben less than tensors'size(" << tensors.size() << ") and larger than 0.";
            }
        }
        auto loc = tensors[0].GetBuffer();
        auto conf = tensors[1].GetBuffer();
        auto shape = tensors[0].GetShape();
        auto iou = tensors[2].GetBuffer();

        cv::Mat PriorBox;
        cv::Mat location = cv::Mat(shape[1], shape[2], CV_32FC1, tensors[0].GetBuffer());
        GeneratePriorBox(PriorBox);

        float width_resize = resizedImageInfos[0].widthResize;
        float height_resize = resizedImageInfos[0].heightResize;
        float width_original = resizedImageInfos[0].widthOriginal;
        float height_original = resizedImageInfos[0].heightOriginal;
        float width_resize_scale = width_resize / width_original;
        float height_resize_scale = height_resize / height_original;
        float resize_scale_factor = 1.0;
        if (width_resize_scale >= height_resize_scale) {
            resize_scale_factor = height_resize_scale;
        } else {
            resize_scale_factor = width_resize_scale;
        }
        cv::Mat res = decode_for_loc(location, PriorBox, resize_scale_factor);

        uint32_t batchSize = shape[0];
        uint32_t VectorNum = shape[1];
        for (uint32_t i = 0; i < batchSize; i++) {
            std::vector <ObjectInfo> objectInfo;
            auto dataPtr_Conf = (float *) tensors[1].GetBuffer() + i * tensors[1].GetByteSize() / batchSize;
            auto dataPtr_Iou = (float *) tensors[2].GetBuffer() + i * tensors[2].GetByteSize() / batchSize;
            for (uint32_t j = 0; j < VectorNum; j++) {
                float* begin_Conf = dataPtr_Conf + j * 2;
                float* begin_Iou = dataPtr_Iou + j;
                float conf = *(begin_Conf + 1);
                float iou = *begin_Iou;
                if (iou < 0.f) iou = 0.f;
                if (iou > 1.f) iou = 1.f;

                conf = sqrtf(iou * conf);
                if (conf> confThresh_) {
                    ObjectInfo objInfo;
                    objInfo.confidence = conf;
                    objInfo.x0 = res.at<float>(j, LEFTTOPX) * IMAGE_WIDTH / width_resize_scale;
                    objInfo.y0 = res.at<float>(j, LEFTTOPY) * IMAGE_HEIGHT / height_resize_scale;
                    objInfo.x1 = res.at<float>(j, RIGHTTOPX) * IMAGE_WIDTH / width_resize_scale;
                    objInfo.y1 = res.at<float>(j, RIGHTTOPY) * IMAGE_HEIGHT / height_resize_scale;

                    objectInfo.push_back(objInfo);
                }
            }
            MxBase::NmsSort(objectInfo, iouThresh_);
            objectInfos.push_back(objectInfo);
        }
        LogInfo << "YunetPostProcess write results successed.";
    }
    APP_ERROR YunetPostProcess::Process(const std::vector<TensorBase> &tensors,
                                        std::vector<std::vector<ObjectInfo>> &objectInfos,
                                        const std::vector<ResizedImageInfo> &resizedImageInfos,
                                        const std::map<std::string, std::shared_ptr<void>> &configParamMap)
    {
        LogInfo << "Start to Process YunetPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << "CheckAndMoveTensors failed. ret=" << ret;
            return ret;
        }
        ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
        LogInfo << "End to Process YunetPostProcess.";
        return APP_ERR_OK;
    }

    /*
     * @description: Generate prior boxes for detection boxes decoding
     * @param anchors  A Matrix used to save prior boxes that contains box coordinates(x0,y0,x1,y1), shape[21824,4]
     */
    void YunetPostProcess::GeneratePriorBox(cv::Mat &anchors)
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
                    for (int l = 0; l < 3 && PRIOR_PARAMETERS[k][l] != -1; l++) {
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
    /*
     * @description: Generate prior boxes for detection boxes decoding
     * @param loc:  The matrix which contains box biases, shape[21824, 4]
     * @param prior: The matrix which contains prior box coordinates, shape[21824,4]
     * @param resize_scale_factor: The factor of min(WidthOriginal/WidthResize, HeightOriginal/HeightResize)
     * @param boxes: The matrix which contains detection box coordinates(x0,y0,x1,y1), shape[21824,4]
     */
    cv::Mat YunetPostProcess::decode_for_loc(cv::Mat &loc, cv::Mat &prior, float resize_scale_factor) {
        cv::Mat loc_first = loc.colRange(0, 2);
        cv::Mat loc_last = loc.colRange(2, 4);
        cv::Mat prior_first = prior.colRange(0, 2);
        cv::Mat prior_last = prior.colRange(2, 4);

        cv::Mat boxes1 = prior_last + (loc_first*VARIANCE[0]).mul(prior_first);
        cv::Mat boxes2;
        cv::exp(loc_last * VARIANCE[1], boxes2);
        boxes2 = boxes2.mul(prior_first);
        boxes1 = boxes1 - boxes2 / DIV_TWO;
        boxes2 = boxes2 + boxes1;

        cv::Mat boxes;
        cv::hconcat(boxes1, boxes2, boxes);
        if (resize_scale_factor == 0) {
            LogError << "resize_scale_factor is 0.";
        }
        return boxes;
    }

    extern "C" {
        std::shared_ptr <MxBase::YunetPostProcess> GetObjectInstance()
        {
            LogInfo << "Begin to get YunetPostProcess instance.";
            auto instance = std::make_shared<MxBase::YunetPostProcess>();
            LogInfo << "End to get YunetPostProcess instance.";
            return instance;
        }
    }
}