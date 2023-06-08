/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef MAIN_MULTIOBJECTTRACKER_H
#define MAIN_MULTIOBJECTTRACKER_H

#include "MxBase/MxBase.h"
#include "MxBase/CV/MultipleObjectTracking/Huangarian.h"
#include "MxBase/CV/MultipleObjectTracking/KalmanTracker.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/Common/Constants.h"
#include "../allobjectStructure.h"

const int FLOAT_TO_INT = 1000;
const int HITS_THREAHOLD = 2;

struct TrackInfo {
    uint32_t trackId;
    uint32_t age;
    uint32_t hits;
    MxBase::TrackFlag trackFlag;
};

struct TrackLet {
    std::string parentName;
    uint32_t memberId;
    TrackInfo tackInfo;
    MxBase::KalmanTracker kalman;
    MxBase::ObjectInfo detecInfo;
    uint32_t lostAge;
};

class MultiObjectTracker {
public:
    void Process(FrameImage &frameImage, std::vector<MxBase::ObjectInfo> &detectedObjectInfos, std::vector<TrackLet> &trackLetList);

    std::pair<FrameImage, MxBase::ObjectInfo> GetTrackLetBuffer(uint32_t trackID);

private:
    std::vector<TrackLet> trackLetList_ = {};
    std::map<uint32_t, std::pair<FrameImage, MxBase::ObjectInfo>> trackLetBuffer_;
    float trackThreshold_ = 0.f;
    uint32_t lostThreshold_ = 0;
    uint32_t method_ = 0;
    uint32_t generatedId_ = 0;

    APP_ERROR matchProcess_(std::vector<MxBase::ObjectInfo> &detectedObjectInfo, std::vector<cv::Point> &matchedTrackedDetected,
                            std::vector<MxBase::ObjectInfo> &unmatchedObjectQueue);

    APP_ERROR trackObjectUpdate_(std::vector<MxBase::ObjectInfo> &detectedObjectInfo, std::vector<cv::Point> &matchedTrackedDetected,
                                 std::vector<MxBase::ObjectInfo> &unmatchedObjectQueue);

    static MxBase::DetectBox ConvertToDetectBox(const MxBase::ObjectInfo &objectInfo);

    float calcSimilarity_(TrackLet &trackLet, MxBase::ObjectInfo &objectInfo);

    void trackObjectPredict_();

    void filterLowThreshold_(const MxBase::HungarianHandle &hungarianHandleObj, const std::vector<std::vector<int>> &disMatrix,
                             std::vector<cv::Point> &matchedTrackedDetected, std::vector<bool> &detectObjectFalgVec);

    void updateTrackLet_(FrameImage &frameImage, const std::vector<cv::Point> &matchedTrackedDetected,
                         std::vector<MxBase::ObjectInfo> &detectedObjectInfos, std::vector<MxBase::ObjectInfo> &unmatchedObjectQueue);

    void updateMatchedTrackLet_(FrameImage &frameImage, const std::vector<cv::Point> &matchedTrackedDetected,
                                std::vector<MxBase::ObjectInfo> &detectedObjectInfos);

    void addNewDetectedObject_(FrameImage &frameImage, std::vector<MxBase::ObjectInfo> &unmatchedObjectQueue);

    void updateLostTrackLet_();

    void updateTrackLetBuffer_(const FrameImage& frameImage, TrackLet& trackLet);
};

#endif /* MAIN_MULTIOBJECTTRACKER_H */
