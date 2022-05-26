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
    
    const float IMAGE_WIDTH = 1920.0;
    const float IMAGE_HEIGHT = 1080.0;
    const float STEPS[4] = {8.0, 16.0, 32.0, 64.0};
    const float VARIANCE[2] = {0.1, 0.2};}
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

void KPYunetPostProcess::KeypointDetectionOutput(const std::vector <TensorBase>& tensors,
                                                 std::vector <std::vector<KeyPointDetectionInfo>>& keypointInfos, const std::vector <ResizedImageInfo>& resizedImageInfos)
    {
        LogInfo << "KPYunetPostProcess start to write results.";
                
        /*for (auto num : { KeypointInfoTensor_, KeypointConfTensor_ }) {
            if ((num >= tensors.size()) || (num < 0)) {
                LogError << GetError(APP_ERR_INVALID_PARAM) << "TENSOR(" << num
                    << ") must ben less than tensors'size(" << tensors.size() << ") and larger than 0.";
            }
        }
        */
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
        }
        else{
            resize_scale_factor = width_resize_scale;        
        }        
        cv::Mat res = decode_for_loc(location, PriorBox, resize_scale_factor); 

        uint32_t batchSize = shape[0];
        uint32_t VectorNum = shape[1];

 
        std::map<ObjectInfo, KeyPointDetectionInfo> match;


        for (uint32_t i = 0; i < batchSize; i++){
            std::vector <ObjectInfo> objectInfo;
            std::vector <ObjectInfo> objectInfoSorted;
            std::vector <KeyPointDetectionInfo> keypointInfo;
            std::vector <KeyPointDetectionInfo> keypointInfoSorted;
            auto dataPtr_Conf = (float *) tensors[1].GetBuffer() + i * tensors[1].GetByteSize() / batchSize;
            auto dataPtr_Iou = (float *) tensors[2].GetBuffer() + i * tensors[2].GetByteSize() / batchSize;
            for (uint32_t j = 0; j < VectorNum; j++) {
                float* begin_Conf = dataPtr_Conf + j*2; 
                float* begin_Iou = dataPtr_Iou + j; 
                float conf = *(begin_Conf + 1);
                float iou = *begin_Iou;
                if(iou < 0.f) iou = 0.f;
                if(iou > 1.f) iou = 1.f;

                conf = sqrtf(iou * conf);
                if(conf> confThresh_ ){
                    ObjectInfo objInfo;
                    KeyPointDetectionInfo kpInfo;

                    objInfo.confidence = conf;
                    objInfo.x0 = res.at<float>(j, LEFTTOPX) * IMAGE_WIDTH / width_resize_scale;
                    objInfo.y0 = res.at<float>(j, LEFTTOPY) * IMAGE_HEIGHT / height_resize_scale;
                    objInfo.x1 = res.at<float>(j, RIGHTTOPX) * IMAGE_WIDTH / width_resize_scale;
                    objInfo.y1 = res.at<float>(j, RIGHTTOPY) * IMAGE_HEIGHT / height_resize_scale;
                    objInfo.classId = 1;
                    
                    kpInfo.score = conf;
                    for(int k = 0; k < 5; k++)
                    {
                        kpInfo.keyPointMap[k].push_back(res.at<float>(j, 4 + k * 2) * IMAGE_WIDTH / width_resize_scale);
                        kpInfo.keyPointMap[k].push_back(res.at<float>(j, 4 + k * 2 + 1) * IMAGE_HEIGHT / height_resize_scale);
                    }
                    objectInfo.push_back(objInfo);
                    keypointInfo.push_back(kpInfo);
                    match[objInfo] = kpInfo;
                }
            }
            MxBase::NmsSort(objectInfo, iouThresh_);

            for (uint32_t j = 0; j < objectInfo.size(); j++){
                keypointInfoSorted.push_back(match[objectInfo[j]]);                
            }

            keypointInfos.push_back(keypointInfoSorted);   
        }      
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
    // 'min_sizes': [[32, 64, 128], [256], [512]], this parameter is used for generating prior box
    // now [10,16,24],[32,48],[64,96],[128,192,256]
    std::vector<int> min_sizes[4];
    min_sizes[0].emplace_back(10);  
    min_sizes[0].emplace_back(16);
    min_sizes[0].emplace_back(24);
    min_sizes[1].emplace_back(32);  
    min_sizes[1].emplace_back(48);
    min_sizes[2].emplace_back(64);
    min_sizes[2].emplace_back(96);  
    min_sizes[3].emplace_back(128);
    min_sizes[3].emplace_back(192);
    min_sizes[3].emplace_back(256);  
    std::vector<std::vector<int>>feature_maps(4, std::vector<int>(2));
    for (int i = 0; i < feature_maps.size(); i++) {
        feature_maps[i][0] = IMAGE_HEIGHT / STEPS[i];
        feature_maps[i][1] = IMAGE_WIDTH / STEPS[i];
    }
    for (int k = 0; k < feature_maps.size(); k++){
        auto f = feature_maps[k];
        auto _min_sizes = min_sizes[k];
        float step = (float)STEPS[k];
        for (int i = 0; i < f[0]; i++){
            for (int j = 0; j < f[1]; j++) {
                for (auto min_size : _min_sizes) {
                    cv::Mat anchor(1, 4, CV_32F);
                    float center_x = (j + 0.5f) * step;
                    float center_y = (i + 0.5f) * step;

                    //anchor.at<float>(0,0) 
                    float xmin = (center_x - (float)min_size / 2.f) / IMAGE_WIDTH;
                    float ymin = (center_y - (float)min_size / 2.f) / IMAGE_HEIGHT;
                    float xmax = (center_x + (float)min_size / 2.f) / IMAGE_WIDTH;
                    float ymax = (center_y + (float)min_size / 2.f) / IMAGE_HEIGHT;

                    float prior_width = xmax - xmin;
                    float prior_height = ymax - ymin;
                    float prior_center_x = (xmin + xmax)/2;
                    float prior_center_y = (ymin + ymax)/2;

                    anchor.at<float>(0,0) = prior_width;
                    anchor.at<float>(0,1) = prior_height;
                    anchor.at<float>(0,2) = prior_center_x;
                    anchor.at<float>(0,3) = prior_center_y;

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
cv::Mat KPYunetPostProcess::decode_for_loc(cv::Mat &loc, cv::Mat &prior, float resize_scale_factor){
    cv::Mat loc_first = loc.colRange(0,2);
    cv::Mat loc_last = loc.colRange(2,4);
    cv::Mat prior_first = prior.colRange(0,2);  //center
    cv::Mat prior_last = prior.colRange(2,4);   //size
    cv::Mat facepoint = loc.colRange(4,14);

    cv::Mat boxes1 = prior_last + (loc_first * VARIANCE[0]).mul(prior_first);
    cv::Mat boxes2;
    cv::exp(loc_last * VARIANCE[1], boxes2);
    boxes2 = boxes2.mul(prior_first);
    boxes1 = boxes1 - boxes2 / 2;
    boxes2 = boxes2 + boxes1;

    cv::Mat boxes3;
    for (int i = 0; i < 5; i++)
    {
        cv::Mat singlepoint = facepoint.colRange(i * 2, (i + 1) * 2);
        singlepoint = prior_last + (singlepoint * VARIANCE[0]).mul(prior_first);
        if (i == 0) boxes3 = singlepoint;
        else cv::hconcat(boxes3, singlepoint, boxes3);
    }
    //lmx = prior_variance[0] * lmx * prior_width + prior_center_x;


    cv::Mat boxes;
    cv::hconcat(boxes1, boxes2, boxes);
    cv::hconcat(boxes, boxes3, boxes);
    if (resize_scale_factor == 0){
        LogError << "resize_scale_factor is 0.";
    }
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