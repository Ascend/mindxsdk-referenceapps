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

#include "multiObjectTracker.h"

std::pair<FrameImage, MxBase::ObjectInfo> MultiObjectTracker::GetTrackLetBuffer(uint32_t trackID)
{
    return trackLetBuffer_[trackID];
}

void MultiObjectTracker::Process(FrameImage &frameImage, std::vector<MxBase::ObjectInfo> &detectedObjectInfos, std::vector<TrackLet> &trackLetList)
{
    std::vector<MxBase::ObjectInfo> unmatchedObjectQueue;
    std::vector<cv::Point> matchedTrackedDetected;

    FrameImage frame;
    frame.image = frameImage.image;
    frame.frameID = frameImage.frameID;
    frame.channelID = frameImage.channelID;

    if (!detectedObjectInfos.empty())
    {
        matchProcess_(detectedObjectInfos, matchedTrackedDetected, unmatchedObjectQueue);
    }
    else
    {
        for (auto &trackLet : trackLetList_)
        {
            trackLet.trackInfo.trackFlag = MxBase::LOST_OBJECT;
        }
    }

    updateTrackLet_(frame, matchedTrackedDetected, detectedObjectInfos, unmatchedObjectQueue);

    for (auto itr = trackLetList_.begin(); itr != trackLetList_.end();)
    {
        if (itr->trackInfo.trackFlag != MxBase::LOST_OBJECT)
        {
            trackLetList.push_back(*itr);
            ++itr;
        }
        else if (itr->lostAge > lostThreshold_)
        {
            trackLetList.push_back(*itr);
            itr = trackLetList_.erase(itr);
        }
        else
        {
            ++itr;
        }
    }
}

APP_ERROR MultiObjectTracker::matchProcess_(std::vector<MxBase::ObjectInfo> &detectedObjectInfo, std::vector<cv::Point> &matchedTrackedDetected,
                                            std::vector<MxBase::ObjectInfo> &unmatchedObjectQueue)
{
    if (!trackLetList_.empty())
    {
        trackObjectPredict_();
        APP_ERROR ret = trackObjectUpdate_(detectedObjectInfo, matchedTrackedDetected, unmatchedObjectQueue);
        if (ret != APP_ERR_OK)
        {
            LogError << GetError(ret) << "trackObjectUpdate_ failed.";
            return ret;
        }
    }
    else
    {
        if (!detectedObjectInfo.empty())
        {
            for (size_t i = 0; i < detectedObjectInfo.size(); ++i)
            {
                unmatchedObjectQueue.push_back(detectedObjectInfo[i]);
            }
        }
    }

    return APP_ERR_OK;
}

APP_ERROR MultiObjectTracker::trackObjectUpdate_(std::vector<MxBase::ObjectInfo> &detectedObjectInfo, std::vector<cv::Point> &matchedTrackedDetected,
                                                 std::vector<MxBase::ObjectInfo> &unmatchedObjectQueue)
{
    std::vector<bool> detectObjectFlagVec;
    for (size_t i = 0; i < detectedObjectInfo.size(); ++i)
    {
        detectObjectFlagVec.push_back(false);
    }

    std::vector<std::vector<int>> simMatrix;
    simMatrix.clear();
    simMatrix.resize(trackLetList_.size(), std::vector<int>(detectedObjectInfo.size(), 0));

    for (size_t i = 0; i < detectedObjectInfo.size(); ++i)
    {
        for (size_t j = 0; j < trackLetList_.size(); ++j)
        {
            float similarity = calcSimilarity_(trackLetList_[j], detectedObjectInfo[i]);
            simMatrix[j][i] = static_cast<int>(similarity * FLOAT_TO_INT);
        }
    }

    MxBase::HungarianHandle hungarianHandleObj;
    APP_ERROR ret = HungarianHandleInit(hungarianHandleObj, trackLetList_.size(), detectedObjectInfo.size());
    if (ret != APP_ERR_OK)
    {
        LogError << GetError(ret) << "HungarianHandleInit failed.";
        return ret;
    }

    HungarianSolve(hungarianHandleObj, simMatrix, trackLetList_.size(), detectedObjectInfo.size());
    filterLowThreshold_(hungarianHandleObj, simMatrix, matchedTrackedDetected, detectObjectFlagVec);
    for (size_t i = 0; i < detectObjectFlagVec.size(); ++i)
    {
        if (!detectObjectFlagVec[i])
        {
            unmatchedObjectQueue.push_back(detectedObjectInfo[i]);
        }
    }
    return APP_ERR_OK;
}

MxBase::DetectBox MultiObjectTracker::ConvertToDetectBox(const MxBase::ObjectInfo &objectInfo)
{
    MxBase::DetectBox detectBox{};
    float height = objectInfo.y1 - objectInfo.y0;
    float width = objectInfo.x1 - objectInfo.x0;
    detectBox.x = objectInfo.x0;
    detectBox.y = objectInfo.y0;
    detectBox.height = height;
    detectBox.width = width;
    detectBox.classID = objectInfo.classId;
    detectBox.prob = objectInfo.confidence;
    return detectBox;
}

float MultiObjectTracker::calcSimilarity_(TrackLet &trackLet, MxBase::ObjectInfo &objectInfo)
{
    float preWidth = trackLet.detecInfo.x1 - trackLet.detecInfo.x0;
    float preHeight = trackLet.detecInfo.y1 - trackLet.detecInfo.y0;
    float curWidth = objectInfo.x1 - objectInfo.x0;
    float curHeight = objectInfo.y1 - objectInfo.y0;
    cv::Rect_<float> preBox(trackLet.detecInfo.x0, trackLet.detecInfo.y0, preWidth, preHeight);
    cv::Rect_<float> curBox(objectInfo.x0, objectInfo.y0, curWidth, curHeight);
    float intersectionArea = (preBox & curBox).area();
    float unionArea = preBox.area() + curBox.area() - intersectionArea;
    if (std::fabs(unionArea) < MxBase::EPSILON)
    {
        return 0.f;
    }
    return intersectionArea / unionArea;
}

void MultiObjectTracker::trackObjectPredict_()
{
    for (auto &trackLet : trackLetList_)
    {
        MxBase::DetectBox detectBox = trackLet.kalman.Predict();
        trackLet.detecInfo.x0 = detectBox.x;
        trackLet.detecInfo.x1 = detectBox.x + detectBox.width;
        trackLet.detecInfo.y0 = detectBox.y;
        trackLet.detecInfo.y1 = detectBox.y + detectBox.height;
    }
}

void MultiObjectTracker::filterLowThreshold_(const MxBase::HungarianHandle &hungarianHandleObj, const std::vector<std::vector<int>> &disMatrix,
                                             std::vector<cv::Point> &matchedTrackedDetected, std::vector<bool> &detectObjectFalgVec)
{
    for (size_t i = 0; i < trackLetList_.size(); ++i)
    {
        if (hungarianHandleObj.resX[i] != -1 &&
            disMatrix[i][hungarianHandleObj.resX[i]] >= (trackThreshold_ * FLOAT_TO_INT))
        {
            matchedTrackedDetected.push_back(cv::Point(i, hungarianHandleObj.resX[i]));
            detectObjectFalgVec[hungarianHandleObj.resX[i]] = true;
        }
        else
        {
            trackLetList_[i].trackInfo.trackFlag = MxBase::LOST_OBJECT;
        }
    }
}

void MultiObjectTracker::updateTrackLet_(FrameImage &frameImage, const std::vector<cv::Point> &matchedTrackedDetected,
                                         std::vector<MxBase::ObjectInfo> &detectedObjectInfos, std::vector<MxBase::ObjectInfo> &unmatchedObjectQueue)
{
    updateMatchedTrackLet_(frameImage, matchedTrackedDetected, detectedObjectInfos);
    addNewDetectedObject_(frameImage, unmatchedObjectQueue);
    updateLostTrackLet_();
}

void MultiObjectTracker::updateMatchedTrackLet_(FrameImage &frameImage, const std::vector<cv::Point> &matchedTrackedDetected,
                                                std::vector<MxBase::ObjectInfo> &detectedObjectInfos)
{
    for (size_t i = 0; i < matchedTrackedDetected.size(); ++i)
    {
        int traceIndex = matchedTrackedDetected[i].x;
        int detectIndex = matchedTrackedDetected[i].y;

        trackLetList_[traceIndex].trackInfo.age++;
        trackLetList_[traceIndex].trackInfo.hits++;
        if (trackLetList_[traceIndex].trackInfo.hits > HITS_THREAHOLD)
        {
            trackLetList_[traceIndex].trackInfo.trackFlag = MxBase::TRACKED_OBJECT;
        }
        trackLetList_[traceIndex].lostAge = 0;
        trackLetList_[traceIndex].detecInfo = detectedObjectInfos[detectIndex];
        MxBase::DetectBox detectBox = ConvertToDetectBox(detectedObjectInfos[detectIndex]);
        trackLetList_[traceIndex].kalman.Update(detectBox);

        updateTrackLetBuffer_(frameImage, trackLetList_[traceIndex]);
    }
}

void MultiObjectTracker::addNewDetectedObject_(FrameImage &frameImage, std::vector<MxBase::ObjectInfo> &unmatchedObjectQueue)
{
    for (auto &detectObject : unmatchedObjectQueue)
    {
        TrackLet trackLet{};
        generatedId_++;
        trackLet.trackInfo.trackId = generatedId_;
        trackLet.trackInfo.age = 1;
        trackLet.trackInfo.hits = 1;
        trackLet.lostAge = 0;
        trackLet.trackInfo.trackFlag = MxBase::NEW_OBJECT;
        trackLet.detecInfo = detectObject;
        MxBase::DetectBox detectBox = ConvertToDetectBox(detectObject);
        trackLet.kalman.CvKalmanInit(detectBox);
        trackLetList_.push_back(trackLet);

        updateTrackLetBuffer_(frameImage, trackLet);
    }
}

void MultiObjectTracker::updateLostTrackLet_()
{
    for (auto &trackLet : trackLetList_)
    {
        if (trackLet.trackInfo.trackFlag == MxBase::LOST_OBJECT)
        {
            trackLet.lostAge++;
            trackLet.trackInfo.age++;
        }
    }
}

void MultiObjectTracker::updateTrackLetBuffer_(const FrameImage &frameImage, TrackLet &trackLet)
{
    auto trackID = trackLet.trackInfo.trackId;
    if (trackLet.trackInfo.trackFlag == MxBase::NEW_OBJECT)
    {
        trackLetBuffer_.insert(std::make_pair(trackID, std::make_pair(frameImage, trackLet.detecInfo)));
        return;
    }

    if (trackLet.trackInfo.trackFlag == MxBase::TRACKED_OBJECT)
    {
        trackLetBuffer_[trackID].first = frameImage;
        trackLetBuffer_[trackID].second = trackLet.detecInfo;
        return;
    }
}