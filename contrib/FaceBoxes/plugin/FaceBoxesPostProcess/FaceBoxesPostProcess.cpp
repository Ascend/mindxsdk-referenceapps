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

#include "FaceBoxesPostProcess.h"
#include "MxBase/Log/Log.h"     


namespace {
    const uint32_t LEFTTOPX = 0;
    const uint32_t LEFTTOPY = 1;
    const uint32_t RIGHTTOPX = 2;
    const uint32_t RIGHTTOPY = 3;
    
    const float IMAGE_SIZE = 1024.0;
    const float STEPS[3] = {32.0, 64.0, 128.0};
    const float VARIANCE[2] = {0.1, 0.2};

    const float EPSILON = 0.00001;
}
namespace MxBase {

FaceboxesPostProcess& FaceboxesPostProcess::operator=(const FaceboxesPostProcess& other)
    {
        if (this == &other) {
            return *this;
        }
        ObjectPostProcessBase::operator=(other);
        return *this;
    }

APP_ERROR FaceboxesPostProcess::Init(const std::map <std::string, std::shared_ptr<void>>& postConfig)
    {
        LogInfo << "Start to Init FaceboxesPostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        LogInfo << "End to Init FaceboxesPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR FaceboxesPostProcess::DeInit()
    {
        return APP_ERR_OK;
    }

void FaceboxesPostProcess::ObjectDetectionOutput(const std::vector <TensorBase>& tensors,
                                                 std::vector <std::vector<ObjectInfo>>& objectInfos, const std::vector <ResizedImageInfo>& resizedImageInfos)
    {
        LogInfo << "FaceboxesPostProcess start to write results.";
                
        for (auto num : { objectInfoTensor_, objectConfTensor_ }) {
            if ((num >= tensors.size()) || (num < 0)) {
                LogError << GetError(APP_ERR_INVALID_PARAM) << "TENSOR(" << num
                    << ") must ben less than tensors'size(" << tensors.size() << ") and larger than 0.";
            }
        }
        auto loc = tensors[0].GetBuffer();
        auto conf = tensors[1].GetBuffer();
        auto shape = tensors[0].GetShape();

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
        cv::Mat res = decode(location, PriorBox, resize_scale_factor); 

        uint32_t batchSize = shape[0];
        uint32_t VectorNum = shape[1];
        for (uint32_t i = 0; i < batchSize; i++){
            std::vector <ObjectInfo> objectInfo;
            auto dataPtr_Conf = (float *) tensors[1].GetBuffer() + i * tensors[1].GetByteSize() / batchSize;
            for (uint32_t j = 0; j < VectorNum; j++) {
                float* begin_Conf = dataPtr_Conf + j*2; 
                if(*(begin_Conf + 1)> confThresh_){
                    ObjectInfo objInfo;
                    objInfo.confidence = *(begin_Conf + 1);
                    objInfo.x0 = res.at<float>(j,LEFTTOPX);
                    objInfo.y0 = res.at<float>(j,LEFTTOPY);
                    objInfo.x1 = res.at<float>(j,RIGHTTOPX);
                    objInfo.y1 = res.at<float>(j,RIGHTTOPY);
                    objectInfo.push_back(objInfo);
                }
            }
            MxBase::NmsSort(objectInfo, iouThresh_);
            objectInfos.push_back(objectInfo);   
        }      
        LogInfo << "FaceboxesPostProcess write results successed.";
}
APP_ERROR FaceboxesPostProcess::Process(const std::vector<TensorBase> &tensors,
                                        std::vector<std::vector<ObjectInfo>> &objectInfos,
                                        const std::vector<ResizedImageInfo> &resizedImageInfos,
                                        const std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "Start to Process FaceboxesPostProcess.";
    APP_ERROR ret = APP_ERR_OK;
    auto inputs = tensors;
    ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed. ret=" << ret;
        return ret;
    }
    ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
    LogInfo << "End to Process FaceboxesPostProcess.";
    return APP_ERR_OK;
}

/*
 * @description: Generate prior boxes for detection boxes decoding
 * @param anchors  A Matrix used to save prior boxes that contains box coordinates(x0,y0,x1,y1), shape[21824,4]
 */
void FaceboxesPostProcess::GeneratePriorBox(cv::Mat &anchors)
 {
 
    std::vector<int> min_sizes[3];
    min_sizes[0].emplace_back(32);
    min_sizes[0].emplace_back(64);
    min_sizes[0].emplace_back(128);
    min_sizes[1].emplace_back(256);
    min_sizes[2].emplace_back(512);
    std::vector<std::vector<int>>feature_maps(3, std::vector<int>(2));

    for (int i = 0; i < feature_maps.size(); i++) {
        feature_maps[i][0] = IMAGE_SIZE / STEPS[i];
        feature_maps[i][1] = IMAGE_SIZE / STEPS[i];
    }
    for (int k = 0; k < feature_maps.size(); k++){
        auto f = feature_maps[k];
        auto _min_sizes = min_sizes[k];
        float step = (float)STEPS[k];
        for (int i = 0; i < f[0]; i++){
            for (int j = 0; j < f[1] ; j++) {
                for (auto min_size : _min_sizes) {
                    cv::Mat anchor(1, 4, CV_32F);
                    anchor.at<float>(0,2) = (float)min_size / IMAGE_SIZE;
                    anchor.at<float>(0,3) = (float)min_size / IMAGE_SIZE;
                    if (min_size == 32){
                        float dense_cx[4];
                        float dense_cy[4];
                        for (int x = 0; x < 4; x++){
                            dense_cx[x] = ((float)j + 0.25 * ((float)x)) * step / IMAGE_SIZE;
                            dense_cy[x] = ((float)i + 0.25 * ((float)x)) * step / IMAGE_SIZE;
                        }
                        for (int m = 0; m < sizeof(dense_cy) / sizeof(dense_cy[0]); m++){
                            for (int n = 0; n < sizeof(dense_cx) / sizeof(dense_cx[0]); n++){
                                anchor.at<float>(0,0) = dense_cx[n];
                                anchor.at<float>(0,1) = dense_cy[m];
                                anchors.push_back(anchor);
                            }
                        }
                    } 
                    else if (min_size == 64){
                        float dense_cx[2];
                        float dense_cy[2];
                        for (int x = 0; x < 2; x++){
                            dense_cx[x] = ((float)j + 0.5 * ((float)x)) * step / IMAGE_SIZE;
                            dense_cy[x] = ((float)i + 0.5 * ((float)x)) * step / IMAGE_SIZE;
                        }
                        for (int m = 0; m < sizeof(dense_cy) / sizeof(dense_cy[0]); m++){
                            for (int n = 0; n < sizeof(dense_cx) / sizeof(dense_cx[0]); n++){
                                anchor.at<float>(0,0) = dense_cx[n];
                                anchor.at<float>(0,1) = dense_cy[m];
                                anchors.push_back(anchor);
                            }
                        }
                    } else{
                        float cx = (((float)j + 0.5) * step) / IMAGE_SIZE;
                        float cy = (((float)i + 0.5) * step) / IMAGE_SIZE;
                        anchor.at<float>(0,0) = cx;
                        anchor.at<float>(0,1) = cy;
                        anchors.push_back(anchor);
                    }   
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
cv::Mat FaceboxesPostProcess::decode(cv::Mat &loc, cv::Mat &prior, float resize_scale_factor){
    cv::Mat loc_first = loc.colRange(0,2);
    cv::Mat loc_last = loc.colRange(2,4);
    cv::Mat prior_first = prior.colRange(0,2);
    cv::Mat prior_last = prior.colRange(2,4);
    cv::Mat boxes1 = prior_first + (loc_first*VARIANCE[0]).mul(prior_last);
    cv::Mat boxes2;
    cv::exp(loc_last * VARIANCE[1], boxes2);
    boxes2 = boxes2.mul(prior_last);
    boxes1 = boxes1 - boxes2 / 2;
    boxes2 = boxes2 + boxes1;

    cv::Mat boxes;
    cv::hconcat(boxes1, boxes2, boxes);
    if ((resize_scale_factor >= -EPSINON) && (resize_scale_factor <= EPSINON)) {
        LogError << "resize_scale_factor is 0.";
    }
    boxes = boxes * IMAGE_SIZE / resize_scale_factor;
    return boxes;

}

extern "C" {
    std::shared_ptr <MxBase::FaceboxesPostProcess> GetObjectInstance()
    {
        LogInfo << "Begin to get FaceboxesPostProcess instance.";
        auto instance = std::make_shared<MxBase::FaceboxesPostProcess>();
        LogInfo << "End to get FaceboxesPostProcess instance.";
        return instance;
    }
}

}