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

#ifndef DBPOSTPROCESS_H
#define DBPOSTPROCESS_H

#include <opencv2/imgproc.hpp>
#include "MxBase/PostProcessBases/TextObjectPostProcessBase.h"
#include "MxBase/ErrorCode/ErrorCode.h"

namespace {
const float THRESH = 0.3;
const float BOXTHRESH = 0.7;
const float UNCLIP_RATIO = 1.6;
const int UNCLIP_DISTANCE = 2;
const uint32_t RESIZED_H = 736;
const uint32_t RESIZED_W = 1312;
const int MAX_CANDIDATES = 999;
const int MIN_SIZE = 3;
const int MAX_VAL = 255;
const int POINT1 = 0;
const int POINT2 = 1;
const int POINT3 = 2;
const int POINT4 = 3;
const int VECTOR_FIFTH_INDEX = 4;
const int POINTNUM = 4;
const int INDEX2 = 2;
const int CURRENT_VERSION = 2000001;
}

namespace MxBase {
class DBPostProcess : public TextObjectPostProcessBase {
public:
    DBPostProcess() = default;

    ~DBPostProcess() = default;

    DBPostProcess(const DBPostProcess &other);

    DBPostProcess &operator = (const DBPostProcess &other);

    /*
     * @description Load the configs and labels from the file.
     * @param labelPath config path and label path.
     * @return APP_ERROR error code.
     */
    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

    /*
     * @description: Do nothing temporarily.
     * @return APP_ERROR error code.
     */
    APP_ERROR DeInit() override;

    APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<TextObjectInfo>> &textObjInfos,
        const std::vector<ResizedImageInfo> &resizedImageInfos = {},
        const std::map<std::string, std::shared_ptr<void>> &configParamMap = {});

    bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

    uint64_t GetCurrentVersion() override
    {
        return CURRENT_VERSION;
    }

private:
    APP_ERROR CheckResizeType(const std::vector<ResizedImageInfo> &resizedImageInfos);

    void ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
        std::vector<std::vector<TextObjectInfo>> &textObjInfos, const std::vector<ResizedImageInfo> &resizedImageInfos);

    void FilterByMinSize(std::vector<cv::Point> &contour, std::vector<cv::Point2f> &box, float &minSide);

    void FilterByBoxScore(const cv::Mat &prediction, std::vector<cv::Point2f> &box, float &score);

    void FilterByClippedMinSize(std::vector<cv::Point2f> &box, float &minSide);

    void ConstructInfo(std::vector<TextObjectInfo> &textObjectInfo, std::vector<cv::Point2f> &box,
        const std::vector<ResizedImageInfo> &resizedImageInfos, const uint32_t &index, float score);

    const int NpClip(const int &coordinate, const int &sideLen);

    float PointsL2Distance(cv::Point2f p1, cv::Point2f p2);

    static bool SortByX(cv::Point2f p1, cv::Point2f p2);

    static bool SortByY(cv::Point2f p1, cv::Point2f p2);

    int minSize_ = MIN_SIZE;
    float thresh_ = THRESH;
    float boxThresh_ = BOXTHRESH;
    float unclipRatio_ = UNCLIP_RATIO;
    uint32_t resizedW_ = RESIZED_W;
    uint32_t resizedH_ = RESIZED_H;
    int candidates_ = MAX_CANDIDATES;
};
#ifdef ENABLE_POST_PROCESS_INSTANCE
extern "C" {
std::shared_ptr<MxBase::DBPostProcess> GetTextObjectInstance();
}
#endif
}

#endif // DBPOSTPROCESS_H
