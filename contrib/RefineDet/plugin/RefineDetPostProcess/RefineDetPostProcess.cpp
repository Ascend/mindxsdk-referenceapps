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

#include "RefineDetPostProcess.h"
#include "MxBase/Log/Log.h"

// Detect_RefineDet(num_classes, self.size, 0, 1000, 0.01, 0.45, 0.01, 500)
// num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh, objectness_thre, keep_top_k
namespace {
    const uint32_t LEFTTOPX = 0;
    const uint32_t LEFTTOPY = 1;
    const uint32_t RIGHTTOPX = 2;
    const uint32_t RIGHTTOPY = 3;
    const int PRIOR_PARAMETERS[4] = {32, 64, 128, 256};
    const float IMAGE_WIDTH = 320.0;
    const float IMAGE_HEIGHT = 320.0;
    const float STEPS[4] = {8.0, 16.0, 32.0, 64.0};
    const float VARIANCE[2] = {0.1, 0.2};
    const uint32_t RECTANGLEPOINT = 4;
    const uint32_t DIM = 2; // image dimension is 2
    const uint32_t RECTANGLE_COLOR = 1;
    const uint32_t KEYPOINT_COLOR = 2;
    const uint32_t HINTPOINT_COLOR = 3;
    const uint32_t DIV_TWO = 2;
}
namespace MxBase {
    RefineDetPostProcess& RefineDetPostProcess::operator=(const RefineDetPostProcess& other)
    {
        if (this == &other) {
            return *this;
        }
        ObjectPostProcessBase::operator=(other);
        return *this;
    }

    APP_ERROR RefineDetPostProcess::Init(const std::map <std::string, std::shared_ptr<void>>& postConfig)
    {
        LogDebug << "Start to Init RefineDetPostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        LogDebug << "End to Init RefineDetPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR RefineDetPostProcess::DeInit()
    {
        return APP_ERR_OK;
    }

    void RefineDetPostProcess::generate_objectInfos(const std::vector <TensorBase>& tensors,
                                                std::vector <std::vector<ObjectInfo>>& objectInfos,
                                                const std::vector <ResizedImageInfo>& resizedImageInfos,
                                                cv::Mat& res)
    {
        auto asm_loc = tensors[0].GetBuffer();
        auto asm_conf = tensors[1].GetBuffer();
        auto odm_loc = tensors[2].GetBuffer();
        auto odm_conf = tensors[3].GetBuffer();
        auto shape = tensors[0].GetShape();
        float width_resize_scale = (float)resizedImageInfos[0].widthResize / resizedImageInfos[0].widthOriginal;
        float height_resize_scale = (float)resizedImageInfos[0].heightResize / resizedImageInfos[0].heightOriginal;
        uint32_t batchSize = shape[0];
        uint32_t VectorNum = shape[1];

        for (uint32_t i = 0; i < batchSize; i++){
            std::vector <ObjectInfo> objectInfo, objectInfoSorted;
            auto asm_dataPtr_Conf = (float *) asm_conf + i * tensors[1].GetByteSize() / batchSize;
            auto odm_dataPtr_Conf = (float *) odm_conf + i * tensors[3].GetByteSize() / batchSize;

            float maxId = 0, maxConf = 0;
            for (uint32_t j = 0; j < VectorNum; j++) {
                float* asm_begin_Conf = asm_dataPtr_Conf + j * 2;
                float* odm_begin_Conf = odm_dataPtr_Conf + j * classNum_; 
                for (int k = 1; k < classNum_; k++)
                {
                    float conf = *(asm_begin_Conf + 1) <= 0.01 ? 0 : *(odm_begin_Conf + k);
                    if(conf > 0.05)
                    {
                        ObjectInfo objInfo;
                        objInfo.confidence = conf;
                        objInfo.classId = k;
                        objInfo.className = configData_.GetClassName(k);
                        objInfo.x0 = res.at<float>(j,LEFTTOPX) * IMAGE_WIDTH / width_resize_scale;
                        objInfo.y0 = res.at<float>(j,LEFTTOPY) * IMAGE_HEIGHT / height_resize_scale;
                        objInfo.x1 = res.at<float>(j,RIGHTTOPX) * IMAGE_WIDTH / width_resize_scale;
                        objInfo.y1 = res.at<float>(j,RIGHTTOPY) * IMAGE_HEIGHT / height_resize_scale;

                        objInfo.x0 = objInfo.x0 > 1 ? objInfo.x0 : 1;
                        objInfo.y0  = objInfo.y0 > 1 ? objInfo.y0 : 1;
                        objInfo.x1 = objInfo.x1 < resizedImageInfos[0].widthOriginal ? objInfo.x1 : resizedImageInfos[0].widthOriginal;
                        objInfo.y1 = objInfo.y1 < resizedImageInfos[0].heightOriginal ? objInfo.y1 : resizedImageInfos[0].heightOriginal;

                        objectInfo.push_back(objInfo);
                    }
                }
            }
            MxBase::NmsSort(objectInfo, iouThresh_);
            objectInfos.push_back(objectInfo);   
        }
    }
    void RefineDetPostProcess::ObjectDetectionOutput(const std::vector <TensorBase>& tensors,
                                                 std::vector <std::vector<ObjectInfo>>& objectInfos,
                                                 const std::vector <ResizedImageInfo>& resizedImageInfos)
    {
        LogDebug << "RefineDetPostProcess start to write results.";
                
        for (auto num : { objectInfoTensor_, objectConfTensor_ }) {
            if ((num >= tensors.size()) || (num < 0)) {
                LogError << GetError(APP_ERR_INVALID_PARAM) << "TENSOR(" << num
                    << ") must ben less than tensors'size(" << tensors.size() << ") and larger than 0.";
            }
        }
        auto asm_loc = tensors[0].GetBuffer();
        auto asm_conf = tensors[1].GetBuffer();
        auto odm_loc = tensors[2].GetBuffer();
        auto odm_conf = tensors[3].GetBuffer();
        auto shape = tensors[2].GetShape();

        cv::Mat PriorBox;
        cv::Mat asm_location = cv::Mat(shape[1], shape[2], CV_32FC1, asm_loc);
        cv::Mat odm_location = cv::Mat(shape[1], shape[2], CV_32FC1, odm_loc);

        GeneratePriorBox(PriorBox);
        cv::Mat res = decode_for_loc(asm_location, PriorBox); 
        res = center_size(res);
        res = decode_for_loc(odm_location, res);

        generate_objectInfos(tensors, objectInfos, resizedImageInfos, res);
        LogDebug << "RefineDetPostProcess write results successed.";
    }
    APP_ERROR RefineDetPostProcess::Process(const std::vector<TensorBase> &tensors,
                                        std::vector<std::vector<ObjectInfo>> &objectInfos,
                                        const std::vector<ResizedImageInfo> &resizedImageInfos,
                                        const std::map<std::string, std::shared_ptr<void>> &configParamMap)
    {
        LogDebug << "Start to Process RefineDetPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << "CheckAndMoveTensors failed. ret=" << ret;
            return ret;
        }
        LogInfo << *(float*)tensors[0].GetBuffer();
        ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
        LogDebug << "End to Process RefineDetPostProcess.";
        return APP_ERR_OK;
    }

    /*
     * @description: Generate prior boxes for detection boxes decoding
     * @param anchors  A Matrix used to save prior boxes that contains box coordinates(x0,y0,x1,y1), shape[21824,4]
     */
    void RefineDetPostProcess::GeneratePriorBox(cv::Mat &anchors)
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
                    int min_size = PRIOR_PARAMETERS[k];
                    cv::Mat anchor(1, 4, CV_32F);
                    float center_x = (j + 0.5f) * step / IMAGE_WIDTH;
                    float center_y = (i + 0.5f) * step / IMAGE_HEIGHT;
                    float step_x = min_size / IMAGE_WIDTH;
                    float step_y = min_size / IMAGE_HEIGHT;

                    anchor.at<float>(0,0) = center_x;
                    anchor.at<float>(0,1) = center_y;
                    anchor.at<float>(0,2) = step_x;
                    anchor.at<float>(0,3) = step_y;
                    anchors.push_back(anchor);

                    anchor.at<float>(0,0) = center_x;
                    anchor.at<float>(0,1) = center_y;
                    anchor.at<float>(0,2) = step_x * sqrtf(2.0);
                    anchor.at<float>(0,3) = step_y / sqrtf(2.0);
                    anchors.push_back(anchor);

                    anchor.at<float>(0,0) = center_x;
                    anchor.at<float>(0,1) = center_y;
                    anchor.at<float>(0,2) = step_x / sqrtf(2.0);
                    anchor.at<float>(0,3) = step_y * sqrtf(2.0);
                    anchors.push_back(anchor);
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

    cv::Mat RefineDetPostProcess::center_size(cv::Mat boxes)
    {
        cv::Mat boxes_first = boxes.colRange(0, 2);
        cv::Mat boxes_last = boxes.colRange(2, 4);
        cv::Mat ret_boxes;
        cv::hconcat((boxes_first + boxes_last) / 2, boxes_last - boxes_first, ret_boxes);    
        return ret_boxes;
    }


    cv::Mat RefineDetPostProcess::decode_for_loc(cv::Mat &loc, cv::Mat &prior) {
        cv::Mat loc_first = loc.colRange(0, 2);
        cv::Mat loc_last = loc.colRange(2, 4);
        cv::Mat prior_first = prior.colRange(0, 2);
        cv::Mat prior_last = prior.colRange(2, 4);

        cv::Mat boxes1 = prior_first + (loc_first * VARIANCE[0]).mul(prior_last);
        cv::Mat boxes2;
        cv::exp(loc_last * VARIANCE[1], boxes2);
        boxes2 = boxes2.mul(prior_last);
        boxes1 = boxes1 - boxes2 / 2;
        boxes2 = boxes2 + boxes1;

        cv::Mat boxes;
        cv::hconcat(boxes1, boxes2, boxes);
        return boxes;
    }

    extern "C" {
        std::shared_ptr <MxBase::RefineDetPostProcess> GetObjectInstance()
        {
            LogInfo << "Begin to get RefineDetPostProcess instance.";
            auto instance = std::make_shared<MxBase::RefineDetPostProcess>();
            LogInfo << "End to get RefineDetPostProcess instance.";
            return instance;
        }
    }
}