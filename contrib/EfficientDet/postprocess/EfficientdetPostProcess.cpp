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

#include "EfficientdetPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include <math.h>
#include <iomanip>
#include<algorithm>

namespace {
    auto g_uint8Deleter = [] (uint8_t *p) { };
}

namespace MxBase {
    EfficientdetPostProcess &EfficientdetPostProcess::operator=(const EfficientdetPostProcess &other) {
        if (this == &other) {
            return *this;
        }
        ObjectPostProcessBase::operator=(other);
        classNum_ = other.classNum_;
        iouThresh_ = other.iouThresh_;
        scoreThresh_ = other.scoreThresh_;
        return *this;
    }

    APP_ERROR EfficientdetPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
        LogInfo << "Start to Init EfficientdetPostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        configData_.GetFileValue<int>("CLASS_NUM", classNum_);
        configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
        configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
        LogInfo << "End to Init EfficientdetPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR EfficientdetPostProcess::DeInit() {
        return APP_ERR_OK;
    }

    bool EfficientdetPostProcess::IsValidTensors(const std::vector<TensorBase> &tensors) const {
        int channel_index = 2;
        int regression_channel_num = 4;
        auto regression_shape = tensors[0].GetShape();
        if (regression_shape[channel_index] != regression_channel_num) {
            LogError << "last dimension of regression tensor is not " << regression_channel_num << ".";
            return false;
        }
        auto classification_shape = tensors[1].GetShape();
        if (classification_shape[channel_index] != classNum_) {
            LogError << "last dimension of classification tensor is not " << classNum_ << ".";
            return false;
        }
        return true;
    }

    /**
     * @brief Parsing MxBase::TensorBase data to regression heatmap and classification heatmap of inference model
     * @param tensors - MxBase::TensorBase vector, regression tensor and classification tensor output from the model
     * inference plugin
     * @param regression - Regression heatmap with parsed data, with shape: [batchsize, boxes_num, (dy, dx, dh, dw)]
     * @param classification - Classification heatmap with parsed data
     * */
    void EfficientdetPostProcess::ReadDataFromTensor(const std::vector <MxBase::TensorBase> &tensors,
                                                     std::vector<std::vector<float> > &regression,
                                                     std::vector<std::vector<float> > &classification) {
        int regressionIdx = 0;
        int classificationIdx = 1;
        // Read regression data
        auto shapeReg = tensors[regressionIdx].GetShape();
        int anchorNum = shapeReg[1];
        int anchorCoorNum = shapeReg[2];
        auto regDataPtr = (uint8_t *)tensors[regressionIdx].GetBuffer();
        std::shared_ptr<void> regressionPointer;
        regressionPointer.reset(regDataPtr, g_uint8Deleter);
        int idx = 0;
        for (int i = 0; i < anchorNum; i++) {
            std::vector<float> detBox {};
            for (int j = 0; j < anchorCoorNum; j++) {
                detBox.push_back(static_cast<float *>(regressionPointer.get())[idx]);
                idx += 1;
            }
            regression.push_back(detBox);
        }

        // Read classification data
        auto shapeClass = tensors[classificationIdx].GetShape();
        int classNum = shapeClass[2];
        auto classDataPtr = (uint8_t *)tensors[classificationIdx].GetBuffer();
        std::shared_ptr<void> classificationPointer;
        classificationPointer.reset(classDataPtr, g_uint8Deleter);
        idx = 0;
        for (int i = 0; i < anchorNum; i++) {
            std::vector<float> detClass {};
            for (int j = 0; j < classNum; j++) {
                detClass.push_back(static_cast<float *>(classificationPointer.get())[idx]);
                idx += 1;
            }
            classification.push_back(detClass);
        }
    }

    /**
     * @brief Generate anchors for the input image
     * @param anchors - Generated anchors, with shape: [batchsize, boxes_num, (y1, x1, y2, x2)]
     * @param netWidth - Width of the model input
     * @param netHeight - Height of the model input
     */
    void EfficientdetPostProcess::GenerateAnchors(std::vector<std::vector<float> > &anchorBoxes,
                                                  const int netWidth, const int netHeight) {
        LogInfo << "Begin to calculate anchor boxes";
        std::vector<std::vector<float>> scaleRatioComb {};
        for (int i = 0; i < scales_.size(); i++) {
            for (int j = 0; j < ratios_.size(); j++) {
                scaleRatioComb.push_back({scales_[i], static_cast<float>(j)});
            }
        };
        // Generate anchors under different stride values
        for (int i = 0; i < strides_.size(); i++) {
            int stride = strides_[i];
            std::vector<float> xCoors {};
            std::vector<float> yCoors {};
            float xValue = stride / 2;
            float yValue = stride / 2;
            while (xValue < netWidth) {
                xCoors.push_back(xValue);
                xValue += stride;
            }
            while (yValue < netHeight) {
                yCoors.push_back(yValue);
                yValue += stride;
            }
            float meshX, meshY;
            for (int k = 0; k < xCoors.size() * yCoors.size(); k++) {
                meshX = xCoors[k % xCoors.size()];
                meshY = yCoors[floor(k / yCoors.size())];
                // Generate anchors under each combination of the scale and the ratio
                for (int j = 0; j < scaleRatioComb.size(); j++) {
                    std::vector<float> scaleRatio = scaleRatioComb[j];
                    float scale = scaleRatio[0];
                    int ratioIndex = static_cast<int>(scaleRatio[1]);
                    float base_anchor_size = anchorScale_ * stride * scale;
                    float anchorSizeX2 = base_anchor_size * ratios_[ratioIndex][0] / 2.0;
                    float anchorSizeY2 = base_anchor_size * ratios_[ratioIndex][1] / 2.0;
                    std::vector<float> box {};
                    // y - anchor_size_y, x - anchor_size_x, y + anchor_size_y, x + anchor_size_x
                    box.push_back(meshY - anchorSizeY2);
                    box.push_back(meshX - anchorSizeX2);
                    box.push_back(meshY + anchorSizeY2);
                    box.push_back(meshX + anchorSizeX2);
                    anchorBoxes.push_back(box);
                }
            }
        }
        LogInfo << "End to calculate anchor boxes";
    }

    /**
     * @brief Convert regression heatmap according to anchors to bounding boxes
     * @param anchors - Anchors calculated according to the input image and preset scale
     * @param regression - Regression heatmap
     * @param transformedAnchors - Calculated bounding boxes
     */
    void EfficientdetPostProcess::RegressBoxes(std::vector<std::vector<float> > &anchors,
                                               std::vector<std::vector<float> > &regression,
                                               std::vector<std::vector<float> > &transformedAnchors) {
        float yCenterAnchor, xCenterAnchor, heightAnchor, widthAnchor;
        float yCenterBox, xCenterBox, heightBox, widthBox;
        for (int i = 0; i < anchors.size(); i++) {
            // center coordinate and size of each anchor
            yCenterAnchor = (anchors[i][0] + anchors[i][2]) / 2;
            xCenterAnchor = (anchors[i][1] + anchors[i][3]) / 2;
            heightAnchor = anchors[i][2] - anchors[i][0];
            widthAnchor = anchors[i][3] - anchors[i][1];
            // center coordinate and size of each regressed bounding box
            heightBox = exp(regression[i][2]) * heightAnchor;
            widthBox = exp(regression[i][3]) * widthAnchor;
            yCenterBox = regression[i][0] * heightAnchor + yCenterAnchor;
            xCenterBox = regression[i][1] * widthAnchor + xCenterAnchor;
            std::vector<float> box {};
            // xmin, ymin, xmax, ymax
            box.push_back(xCenterBox - widthBox / 2);
            box.push_back(yCenterBox - heightBox / 2);
            box.push_back(xCenterBox + widthBox / 2);
            box.push_back(yCenterBox + heightBox / 2);
            transformedAnchors.push_back(box);
        }
    }

    /**
     * @brief Crop the generated bounding box to prevent it from exceeding the boundary of input image
     * @param boxes - Pending bounding boxes
     * @param netWidth - Width of the model input
     * @param netHeight - Height of the model input
     */
    void EfficientdetPostProcess::ClipBoxes(std::vector<std::vector<float> > &boxes,
                                            const int netWidth, const int netHeight) {
        float xMin = 0, yMin = 0;
        float xMax = netWidth, yMax = netHeight;
        // Limit the x coordinate of each bounding box to [0, netWidth]
        // Limit the y coordinate of each bounding box to [0, netHeight]
        for (int i = 0; i < boxes.size(); i++) {
            if (boxes[i][0] < xMin) {
                boxes[i][0] = xMin;
            }
            if (boxes[i][1] < yMin) {
                boxes[i][1] = yMin;
            }
            if (boxes[i][2] > xMax - 1) {
                boxes[i][2] = xMax - 1;
            }
            if (boxes[i][3] > yMax - 1) {
                boxes[i][3] = yMax - 1;
            }
        }
    }

    /**
     * @brief Overall process to generate bounding boxes from regression heatmap and classification heatmap
     * @param anchors - Anchors calculated according to the input image and preset scale
     * @param regression - Regression heatmap
     * @param classification - Classification heatmap
     * @param imageInfo - Image information obtained from the mxpi_imageresize plugin (including the
     * width and height of the original image and those of the zoomed image
     * @param detBoxes - Vector of MxBase::ObjectInfo, which stores the information of detected
     * bounding boxes of one image
     */
    void EfficientdetPostProcess::GenerateBoxes(std::vector<std::vector<float> > &anchors,
                                                std::vector<std::vector<float> > &regression,
                                                std::vector<std::vector<float> > &classification,
                                                ResizedImageInfo imageInfo,
                                                std::vector <MxBase::ObjectInfo> &detBoxes) {
        // Get image size information of model input
        int widthResize = imageInfo.widthResize;
        int heightResize = imageInfo.heightResize;
        int widthOriginal = imageInfo.widthOriginal;
        int heightOriginal = imageInfo.heightOriginal;
        int widthResizeBeforePadding, heightResizeBeforePadding;
        if (widthOriginal > heightOriginal) {
            widthResizeBeforePadding = widthResize;
            heightResizeBeforePadding = static_cast<int>(static_cast<float>(widthResize) /
                    widthOriginal * heightOriginal);
        }
        else {
            heightResizeBeforePadding = heightResize;
            widthResizeBeforePadding = static_cast<int>(static_cast<float>(heightResize) /
                    heightOriginal * widthOriginal);
        }
        std::vector<std::vector<float> > transformedAnchors {};
        // Generate bounding boxes from regression heatmap according to anchors
        RegressBoxes(anchors, regression, transformedAnchors);
        // Clip bounding boxes
        ClipBoxes(transformedAnchors, widthResize, heightResize);
        // Generate a MxBase::ObjectInfo object for each bounding box whose confidence exceeds the threshold
        std::vector<int> keepIndex {};
        float widthRatio = static_cast<float>(widthResizeBeforePadding) / static_cast<float>(widthOriginal);
        float heightRatio = static_cast<float>(heightResizeBeforePadding) / static_cast<float>(heightOriginal);
        for (int i = 0; i < classification.size(); i++) {
            std::vector<float> scoresPerBox = classification[i];
            int maxElementIndex = std::max_element(scoresPerBox.begin(), scoresPerBox.end()) - scoresPerBox.begin();
            float maxElement = *std::max_element(scoresPerBox.begin(), scoresPerBox.end());
            if (maxElement > scoreThresh_) {
                MxBase::ObjectInfo det;
                det.x0 = transformedAnchors[i][0] / widthRatio;
                det.x1 = transformedAnchors[i][2] / widthRatio;
                det.y0 = transformedAnchors[i][1] / heightRatio;
                det.y1 = transformedAnchors[i][3] / heightRatio;
                det.classId = maxElementIndex;
                det.confidence = maxElement;
                detBoxes.emplace_back(det);
            }
        }
    }

    /**
     * @brief Obtain the detection bboxes by post-processing the inference result of the object detection model
     * @param tensors - Regression tensor and classification tensor output from the model inference plugin
     * @param resizedImageInfos - Image information obtained from the mxpi_imageresize plugin (including the
     * width and height of the original image and those of the zoomed image
     * @param objectInfos - Vector of vector of MxBase::ObjectInfo, which stores the information of detected
     * bounding boxes of each input image
     * */
    void EfficientdetPostProcess::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                                        std::vector<std::vector<ObjectInfo>> &objectInfos,
                                                        const std::vector<ResizedImageInfo> &resizedImageInfos) {
        LogInfo << "EfficientdetPostProcess start to write results.";
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
            // Generate Anchors
            int widthResize = resizedImageInfos[i].widthResize;
            int heightResize = resizedImageInfos[i].heightResize;
            std::vector<std::vector<float> > anchors {};
            GenerateAnchors(anchors, widthResize, heightResize);
            // Read data from tensor pointer
            std::vector<std::vector<float> > regression {};
            std::vector<std::vector<float> > classification {};
            ReadDataFromTensor(tensors, regression, classification);
            // generate bounding boxes
            std::vector<ObjectInfo> objectInfo;
            GenerateBoxes(anchors, regression, classification, resizedImageInfos[i], objectInfo);
            MxBase::NmsSort(objectInfo, iouThresh_);
            objectInfos.push_back(objectInfo);
        }
        LogInfo << "EfficientdetPostProcess write results successed.";
    }

    APP_ERROR EfficientdetPostProcess::Process(const std::vector<TensorBase> &tensors,
                                               std::vector<std::vector<ObjectInfo>> &objectInfos,
                                               const std::vector<ResizedImageInfo> &resizedImageInfos,
                                               const std::map<std::string, std::shared_ptr<void>> &paramMap) {
        LogInfo << "Start to Process EfficientdetPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        if (resizedImageInfos.size() == 0) {
            ret = APP_ERR_INPUT_NOT_MATCH;
            LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary "
                                         "for EfficientdetPostProcess.";
            return ret;
        }
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "CheckAndMoveTensors failed.";
            return ret;
        }
        ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
        LogInfo << "End to Process EfficientdetPostProcess.";
        return APP_ERR_OK;
    }


    extern "C" {
    std::shared_ptr<MxBase::EfficientdetPostProcess> GetObjectInstance() {
        LogInfo << "Begin to get EfficientdetPostProcess instance.";
        auto instance = std::make_shared<MxBase::EfficientdetPostProcess>();
        LogInfo << "End to get EfficientdetPostProcess instance.";
        return instance;
    }
    }
}
