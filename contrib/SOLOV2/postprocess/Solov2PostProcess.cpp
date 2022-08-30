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

#include "Solov2PostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include <math.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <random>

namespace {
    auto g_uint8Deleter = [] (uint8_t *p) { };
}

namespace MxBase {
    Solov2PostProcess::Solov2PostProcess(const Solov2PostProcess &other) {
        classNum_ = other.classNum_;
        scoreThresh_ = other.scoreThresh_;
        height_ = other.height_;
        width_ = other.width_;
    }

    Solov2PostProcess &Solov2PostProcess::operator=(const Solov2PostProcess &other) {
        if (this == &other) {
            return *this;
        }
        ObjectPostProcessBase::operator=(other);
        classNum_ = other.classNum_;
        scoreThresh_ = other.scoreThresh_;
        height_ = other.height_;
        width_ = other.width_;
        return *this;
    }

    APP_ERROR Solov2PostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
        LogInfo << "Start to Init Solov2PostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        configData_.GetFileValue<int>("CLASS_NUM", classNum_);
        configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
        configData_.GetFileValue<int>("HEIGHT", height_);
        configData_.GetFileValue<int>("WIDTH", width_);
        LogInfo << "End to Init Solov2PostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR Solov2PostProcess::DeInit() {
        return APP_ERR_OK;
    }

    void Solov2PostProcess::ReadDataFromTensor(const std::vector <MxBase::TensorBase> &tensors,
                                               std::vector<std::vector<std::vector<uint8_t>>> &seg,
                                               std::vector<int> &label, std::vector<float> &score) {
        int div = 2;
        int pad_left = (width_ - img_w_) / div;  //  divide the pad into left and right.
        int pad_top = (height_ - img_h_) / div;  //  divide the pad into top and bottom.
        // Read regression data
        auto shapeSeg = tensors[0].GetShape();
        auto segDataPtr = (uint8_t *)tensors[0].GetBuffer();
        std::shared_ptr<void> segPointer;
        segPointer.reset(segDataPtr, g_uint8Deleter);
        int idx = 0;
        int div_seg = 4;
        int tmp_width = width_ / div_seg;
        int tmp_height = height_ / div_seg;
        int num_seg = 100;
        float half = 0.5;
        for (int i = 0; i < num_seg; i++) {
            float tmp_seg[tmp_height][tmp_width];
            std::vector<uint8_t> tmp_ori;
            for (int j = 0; j < tmp_height; j++) {
                for (int k = 0; k < tmp_width; k++) {
                    tmp_seg[j][k] = (static_cast<float *>(segPointer.get())[idx]);
                    tmp_ori.push_back(static_cast<uint8_t *>(segPointer.get())[idx]);
                    idx += 1;
                }
            }
            cv::Mat tmp(tmp_height, tmp_width, CV_32FC1, tmp_seg);
            cv::Mat resize_seg1, resize_seg2;
            cv::resize(tmp, resize_seg1, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);
            cv::Mat tmp2 = resize_seg1(cv::Rect(pad_left, pad_top, img_w_, img_h_));
            cv::resize(tmp2, resize_seg2, cv::Size(ori_w_, ori_h_), 0, 0, cv::INTER_LINEAR);
            std::vector<float> tmp3 = resize_seg2.reshape(1, 1);
            for (int n = 0; n < tmp3.size(); n++) {
                tmp3[n] = tmp3[n] > half ? 1 : 0;
            }
            std::vector<std::vector<uint8_t>> mask_height;
            int idx2 = 0;
            for (int j = 0; j < ori_h_; j++) {
                std::vector<uint8_t> mask_width;
                for (int k = 0; k < ori_w_; k++) {
                    mask_width.push_back(static_cast<int>(tmp3[idx2++]));
                }
                mask_height.push_back(mask_width);
            }
            seg.push_back(mask_height);
        }

        auto shapeLabel = tensors[1].GetShape();
        auto labelDataPtr = (uint8_t *)tensors[1].GetBuffer();
        std::shared_ptr<void> labelPointer;
        labelPointer.reset(labelDataPtr, g_uint8Deleter);
        idx = 0;
        for (int i = 0; i < num_seg; i++) {
            label.push_back(static_cast<int *>(labelPointer.get())[idx]);
            idx += 1;
        }

        auto shapescore = tensors[2].GetShape();
        auto scoreDataPtr = (uint8_t *)tensors[2].GetBuffer();
        std::shared_ptr<void> scorePointer;
        scorePointer.reset(scoreDataPtr, g_uint8Deleter);
        idx = 0;
        for (int i = 0; i < num_seg; i++) {
            score.push_back(static_cast<float *>(scorePointer.get())[idx]);
            idx += 1;
        }
    }


    void Solov2PostProcess::GenerateBoxes(std::vector<std::vector<std::vector<uint8_t>>> &seg,
                                          std::vector<int> &label, std::vector<float> &score,
                                          std::vector <MxBase::ObjectInfo> &detBoxes) {
        // Get image size information of model input
        int blank = classNum_ + 1;
        for (int i = 0; i < seg.size(); i++) {
            MxBase::ObjectInfo det;
            if (i == 0 && score[i] < scoreThresh_) {
                det.classId = blank;
                det.confidence = score[i];
                detBoxes.emplace_back(det);
                break;
            }
            if (score[i] > scoreThresh_) {
                det.classId = label[i];
                det.confidence = score[i];
                det.className = configData_.GetClassName(label[i]);
                det.mask = seg[i];
                detBoxes.emplace_back(det);
            }
        }
    }

    void Solov2PostProcess::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                                  std::vector<std::vector<ObjectInfo>> &objectInfos,
                                                  const std::vector<ResizedImageInfo> &resizedImageInfos) {
        LogInfo << "Solov2PostProcess start to write results.";
        ResizedImageInfo resizedInfo = resizedImageInfos[0];
        if (tensors.size() == 0) {
            return;
        }
        auto shape = tensors[0].GetShape();
        if (shape.size() == 0) {
            return;
        }
        uint32_t batchSize = shape[0];
        for (uint32_t i = 0; i < batchSize; i++) {
            ori_w_ = resizedImageInfos[i].widthOriginal;
            ori_h_ = resizedImageInfos[i].heightOriginal;
            float ratio_h = height_ / static_cast<float>(ori_h_);
            float ratio_w = width_ / static_cast<float>(ori_w_);
            float ratio = std::min(ratio_h, ratio_w);
            img_h_ = int(std::floor(ori_h_ * ratio));
            img_w_ = int(std::floor(ori_w_ * ratio));
            std::vector<std::vector<uint8_t>> seg_ori;
            std::vector<float> score;
            std::vector<int> label;
            std::vector<std::vector<std::vector<uint8_t>>> seg;

            // Read data from tensor pointer
            ReadDataFromTensor(tensors, seg, label, score);
            std::vector<ObjectInfo> objectInfo;
            GenerateBoxes(seg, label, score, objectInfo);
            objectInfos.push_back(objectInfo);
        }
        LogInfo << "Solov2PostProcess write results successed.";
    }

    APP_ERROR Solov2PostProcess::Process(const std::vector<TensorBase> &tensors,
                                         std::vector<std::vector<ObjectInfo>> &objectInfos,
                                         const std::vector<ResizedImageInfo> &resizedImageInfos,
                                         const std::map<std::string, std::shared_ptr<void>> &paramMap) {
        LogInfo << "Start to Process Solov2PostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        if (resizedImageInfos.size() == 0) {
            ret = APP_ERR_INPUT_NOT_MATCH;
            LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary "
                                         "for Solov2PostProcess.";
            return ret;
        }
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "CheckAndMoveTensors failed.";
            return ret;
        }
        ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
        LogInfo << "End to Process Solov2PostProcess.";
        return APP_ERR_OK;
    }


    extern "C" {
    std::shared_ptr<MxBase::Solov2PostProcess> GetObjectInstance() {
        LogInfo << "Begin to get Solov2PostProcess instance.";
        auto instance = std::make_shared<MxBase::Solov2PostProcess>();
        LogInfo << "End to get Solov2PostProcess instance.";
        return instance;
    }
    }
}
