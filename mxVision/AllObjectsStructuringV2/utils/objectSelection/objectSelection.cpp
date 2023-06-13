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

#include "objectSelection.h"

APP_ERROR ObjectSelector::Process(MultiObjectTracker &tracker, std::vector<TrackLet> &trackLetList,
                                  std::vector<std::pair<FrameImage, MxBase::ObjectInfo>> &selectedObjectVec)
{
    for (size_t i = 0; i < trackLetList.size(); i++)
    {
        if (trackLetList[i].trackInfo.trackFlag == MxBase::LOST_OBJECT)
        {
            auto trackID = trackLetList[i].trackInfo.trackId;
            LogInfo << "Object track id: " << trackID << " lost, need to process it.";
            std::pair<FrameImage, MxBase::ObjectInfo> trackLetRes = tracker.GetTrackLetBuffer(trackID);
            LogInfo << "Get trackLet buffer, frame id is " << trackLetRes.first.frameID;
            selectedObjectVec.emplace_back(trackLetRes);
        }
    }

    return APP_ERR_OK;
}