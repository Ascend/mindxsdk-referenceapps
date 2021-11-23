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

#ifndef EFFICIENTDET_POST_PROCESS_H
#define EFFICIENTDET_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include <math.h>

namespace {
const int REGRESSION_CHANNEL_NUM = 4;
const int DEFAULT_CLASS_NUM = 90;
const int DEFAULT_ANCHOR_SCALE = 4;
const float DEFAULT_SCORE_THRESH = 0.2;
const float DEFAULT_IOU_THRESH = 0.2;
const std::vector<float> DEFAULT_SCALES = {pow(2, 0), pow(2, 1.0 / 3.0), pow(2, 2.0 / 3.0)};
const std::vector<int> DEFAULT_STRIDES = {int(pow(2, 3)), int(pow(2, 4)), int(pow(2, 5)), int(pow(2, 6)), int(pow(2, 7))};
const std::vector<std::vector<float>> DEFAULT_RATIOS = {{1.0, 1.0}, {1.4, 0.7}, {0.7, 1.4}};
}

namespace MxBase {
    class EfficientdetPostProcess: public ObjectPostProcessBase{

    public:
        EfficientdetPostProcess() = default;

        ~EfficientdetPostProcess() = default;

        EfficientdetPostProcess(const EfficientdetPostProcess &other) = default;

        EfficientdetPostProcess &operator=(const EfficientdetPostProcess &other);

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>> &postConfig);

        APP_ERROR DeInit();

        APP_ERROR Process(const std::vector <MxBase::TensorBase> &tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {},
                          const std::map <std::string, std::shared_ptr<void>> &paramMap = {});

    protected:
        bool IsValidTensors(const std::vector <MxBase::TensorBase> &tensors) const;

        /**
         * @brief Obtain the detection bboxes by post-processing the inference result of the object detection model
         * @param tensors - Regression tensor and classification tensor output from the model inference plugin
         * @param resizedImageInfos - Image information obtained from the mxpi_imageresize plugin (including the
         * width and height of the original image and those of the zoomed image
         * @param objectInfos - Vector of vector of MxBase::ObjectInfo, which stores the information of detected
         * bounding boxes of each input image
         * */
        void ObjectDetectionOutput(const std::vector <MxBase::TensorBase> &tensors,
                                   std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                                   const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {});

        /**
         * @brief Parsing MxBase::TensorBase data to regression heatmap and classification heatmap of inference model
         * @param tensors - MxBase::TensorBase vector, regression tensor and classification tensor output from the model
         * inference plugin
         * @param regression - Regression heatmap with parsed data, with shape: [batchsize, boxes_num, (dy, dx, dh, dw)]
         * @param classification - Classification heatmap with parsed data
         * */
        void ReadDataFromTensor(const std::vector <MxBase::TensorBase> &tensors,
                                std::vector<std::vector<float> > &regression,
                                std::vector<std::vector<float> > &classification);

        /**
         * @brief Generate anchors for the input image
         * @param anchors - Generated anchors, with shape: [batchsize, boxes_num, (y1, x1, y2, x2)]
         * @param netWidth - Width of the model input
         * @param netHeight - Height of the model input
         */
        void GenerateAnchors(std::vector<std::vector<float> > &anchors,
                          const int netWidth, const int netHeight);

        /**
         * @brief Convert regression heatmap according to anchors to bounding boxes
         * @param anchors - Anchors calculated according to the input image and preset scale
         * @param regression - Regression heatmap
         * @param transformedAnchors - Calculated bounding boxes
         */
        void RegressBoxes(std::vector<std::vector<float> > &anchors,
                          std::vector<std::vector<float> > &regression,
                          std::vector<std::vector<float> > &transformedAnchors);

        /**
         * @brief Crop the generated bounding box to prevent it from exceeding the boundary of input image
         * @param boxes - Pending bounding boxes
         * @param netWidth - Width of the model input
         * @param netHeight - Height of the model input
         */
        void ClipBoxes(std::vector<std::vector<float> > &boxes,
                       const int netWidth, const int netHeight);

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
        void GenerateBoxes(std::vector<std::vector<float> > &anchors,
                           std::vector<std::vector<float> > &regression,
                           std::vector<std::vector<float> > &classification,
                           ResizedImageInfo imageInfo,
                           std::vector <MxBase::ObjectInfo> &detBoxes);


    protected:
        int anchorScale_ = DEFAULT_ANCHOR_SCALE;
        int classNum_ = DEFAULT_CLASS_NUM;
        float scoreThresh_ = DEFAULT_SCORE_THRESH; // Confidence threhold
        float iouThresh_ = DEFAULT_IOU_THRESH; // Non-Maximum Suppression threshold
        std::vector<int> strides_ = DEFAULT_STRIDES;
        std::vector<float> scales_ = DEFAULT_SCALES;
        std::vector<std::vector<float>> ratios_ = DEFAULT_RATIOS;

    };
    extern "C" {
    std::shared_ptr<MxBase::EfficientdetPostProcess> GetObjectInstance();
    }
}
#endif