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
 
#include "PicodetPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace MxBase {
    PicodetPostProcess &PicodetPostProcess::operator=(const PicodetPostProcess &other)
    {
        if (this == &other) {
            return *this;
        }
        ObjectPostProcessBase::operator=(other);
        scoreThresh_ = other.scoreThresh_;
        nmsThresh_ = other.nmsThresh_;
        classNum_ = other.classNum_;
        stridesNum_ = other.stridesNum_;
        strides_ = other.strides_;
        return *this;
    }

    APP_ERROR PicodetPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig)
    {
        LogDebug << "Start to Init PicodetPostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }

        std::string str;
        configData_.GetFileValue<std::string>("STRIDES", str);
        configData_.GetFileValue<uint32_t>("STRIDES_NUM", stridesNum_);
        configData_.GetFileValue<uint32_t>("CLASS_NUM", classNum_);
        configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
        configData_.GetFileValue<float>("NMS_THRESH", nmsThresh_);
        ret = GetStrides(str);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to get strides.";
            return ret;
        }
        LogDebug << "End to Init PicodetPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR PicodetPostProcess::DeInit()
    {
        return APP_ERR_OK;
    }

    bool PicodetPostProcess::IsValidTensors(const std::vector<TensorBase> &tensors) const
    {
        if (tensors.size() != TENSOR_SIZE) {
            LogError << GetError(APP_ERR_COMM_INVALID_PARAM)  << "The number of tensors (" << tensors.size() << ") "
                    << "is unequal to 8";
            return false;
        }
        for (uint32_t i = 0; i < tensors.size() / 2; i++) {
            auto shape1 = tensors[i].GetShape();
            auto shape2 = tensors[i + stridesNum_].GetShape();
            if (shape1.size() != TENSOR_SHAPE_SIZE || shape2.size() != TENSOR_SHAPE_SIZE) {
                LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "The shape size of tensors (" << shape1.size()
                        << " or " << shape2.size() << ") is unequal to 3.";
                return false;
            }
            if (shape1[3] != classNum_ || shape2[3] != BOXES_TENSOR) {
                LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "The classNum (" << shape1[3] << ") is unequal to "
                        << classNum_ << " or the value of tensor shape " << shape2[3] << ") is unequal to 32.";
                return false;
            }
        }
        return true;
    }

    APP_ERROR PicodetPostProcess::GetStrides(std::string &strStrides)
    {
        if (stridesNum_ <= 0) {
            LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "Failed to get stridesNum (" << stridesNum_ << ").";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        strides_.clear();
        int i = 0;
        int num = strStrides.find(",");
        while (num >= 0 && i < stridesNum_) {
            std::string tmp = strStrides.substr(0, num);
            num++;
            strStrides = strStrides.substr(num, strStrides.size());
            strides_.push_back(stof(tmp));
            i++;
            num = strStrides.find(",");
        }
        if (i != stridesNum_ - 1 || strStrides.size() <= 0) {
            LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "stridesNum (" << stridesNum_
                     << ") is not equal to total number of strides (" << strStrides << ").";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        strides_.push_back(stof(strStrides));
        return APP_ERR_OK;
    }
    
    void PicodetPostProcess::GenerateBbox(const float *&bboxInfo, std::pair<int, int> center, int stride,
                                          const ResizedImageInfo &resizedImageInfo,
                                          ObjectInfo &objectInfo)
    {
        std::vector<float> disPred;
        disPred.resize(BBOX_SIZE);
        for (int i = 0; i < BBOX_SIZE; i++) {
            float dis = 0;
            std::vector<float> disSoftmax;
            for (int j = 0; j < (REG_MAX + 1); j++) {
                disSoftmax.push_back(*(bboxInfo + i * (REG_MAX + 1) + j));
            }
            fastmath::softmax(disSoftmax);
            for (int j = 0; j < REG_MAX + 1; j++) {
                dis += j * disSoftmax[j];
            }
            dis *= stride;
            disPred[i] = dis;
        }

        float resizeX0 = (std::max)(center.first - disPred[0], .0f);
        float resizeY0 = (std::max)(center.second - disPred[1], .0f);
        float resizeX1 = (std::min)(center.first + disPred[2], (float)resizedImageInfo.widthResize);
        float resizeY1  = (std::min)(center.second + disPred[3], (float)resizedImageInfo.heightResize);

        objectInfo.x0 = (float)(resizedImageInfo.widthOriginal * resizeX0) / resizedImageInfo.widthResize;
        objectInfo.y0 = (float)(resizedImageInfo.heightOriginal * resizeY0) / resizedImageInfo.heightResize;
        objectInfo.x1 =(float)(resizedImageInfo.widthOriginal * resizeX1) / resizedImageInfo.widthResize;
        objectInfo.y1  = (float)(resizedImageInfo.heightOriginal * resizeY1) / resizedImageInfo.heightResize;
    }

    void PicodetPostProcess::GetScoreAndLabel(const float *outBuffer, const uint32_t idx,float &score, int &curLabel)
    {
        const float * scores = outBuffer + (idx * classNum_);
        for (uint32_t label = 0; label < classNum_; label++) {
            if (scores[label] > score) {
                score = scores[label];
                curLabel = label;
            }
        }
    }

    APP_ERROR PicodetPostProcess::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                                  std::vector<std::vector<ObjectInfo>> &objectInfos,
                                                  const std::vector<ResizedImageInfo> &resizedImageInfos)
    {
        LogDebug << "PicodetPostProcess start to write results.";
        if (tensors.size() == 0) {
            LogError << "empty tensor";
            return APP_ERR_COMM_NO_EXIST;
        }

        objectInfos.resize(classNum_);
        for (uint32_t i = 0; i < strides_.size(); i++) {
            uint32_t featureHeight = std::ceil((float)resizedImageInfos[0].heightResize /  strides_[i]);
            uint32_t featureWidth = std::ceil((float)resizedImageInfos[0].widthResize /  strides_[i]);
            auto outBuffer1 = (float *)tensors[i].GetBuffer();
            auto outBuffer2 = (float *)tensors[i + strides_.size()].GetBuffer();
            for (uint32_t idx = 0; idx < featureHeight * featureWidth; idx++) {
                float score = 0;
                int curLabel = 0;
                GetScoreAndLabel(outBuffer1, idx, score, curLabel);

                float centerY = ((idx / featureWidth) + 0.5) *  strides_[i];
                float centerX = ((idx % featureWidth) + 0.5) *  strides_[i];
                std::pair<int, int> center(centerX, centerY);

                if (score > scoreThresh_) {
                    ObjectInfo objectInfo;
                    objectInfo.confidence = score;
                    objectInfo.classId = (float)curLabel;
                    objectInfo.className = configData_.GetClassName(curLabel);
                    const float *bboxInfo = outBuffer2 + (idx * BBOX_SIZE * (REG_MAX + 1));
                    GenerateBbox(bboxInfo, center,  strides_[i], resizedImageInfos[0], objectInfo);
                    objectInfos[curLabel].push_back(objectInfo);
                }
            }
        }

        for (int i = 0; i < objectInfos.size(); i++) {
            NmsSort(objectInfos[i], nmsThresh_);
        }

        return APP_ERR_OK;
    }

    APP_ERROR PicodetPostProcess::Process(const std::vector<TensorBase> &tensors,
                                         std::vector<std::vector<ObjectInfo>> &objectInfos,
                                         const std::vector<ResizedImageInfo> &resizedImageInfos,
                                         const std::map<std::string, std::shared_ptr<void>> &paramMap)
    {
        LogDebug << "Start to Process PicodetPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        if (resizedImageInfos.size() == 0) {
            ret = APP_ERR_INPUT_NOT_MATCH;
            LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary for PicodetPostProcess.";
            return ret;
        }
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "CheckAndMoveTensors failed.";
            return ret;
        }
        ret = ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "ObjectDetectionOutput failed.";
            return ret;
        }
        LogDebug << "End to Process PicodetPostProcess.";
        return APP_ERR_OK;
    }


    extern "C" {
    std::shared_ptr <MxBase::PicodetPostProcess> GetObjectInstance()
    {
        LogInfo << "Begin to get PicodetPostProcess instance.";
        auto instance = std::make_shared<MxBase::PicodetPostProcess>();
        LogInfo << "End to get PicodetPostProcess instance.";
        return instance;
    }
    }
}