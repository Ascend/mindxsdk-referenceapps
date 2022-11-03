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

#include "BlockingMap.h"

namespace MxPlugins
{
    BlockingMap::BlockingMap() {}

    BlockingMap::~BlockingMap() {}

    void BlockingMap::Insert(const uint32_t &id, MxBase::MemoryData newData)
    {
        // add std::lock_guard
        std::lock_guard<std::mutex> guard(mtx_);
        // get current timestamp
        using Time = std::chrono::high_resolution_clock;
        auto currentTime = Time::now();
        std::pair<std::chrono::high_resolution_clock::time_point,
                  std::shared_ptr<MxTools::MxpiVisionList>>
            time_MxpiVisionList;
        // set MxpiVisionInfo and MxpiVisionData
        auto mxpiVisionList = copyList(newData);
        time_MxpiVisionList = std::make_pair(currentTime, mxpiVisionList);
        blockingMap_[id] = time_MxpiVisionList;
        keys_.insert(id);
    }

    void BlockingMap::Update(const uint32_t &id, MxBase::MemoryData newData)
    {
        // add std::lock_guard
        std::lock_guard<std::mutex> guard(mtx_);
        std::pair<std::chrono::high_resolution_clock::time_point, std::shared_ptr<MxTools::MxpiVisionList>>
            time_MxpiVisionList = blockingMap_[id];
        auto mxpiVisionList = time_MxpiVisionList.second;
        // set MxpiVisionInfo and MxpiVisionData
        MxTools::MxpiVision *dstMxpivision = mxpiVisionList->add_visionvec();
        MxTools::MxpiVisionInfo *mxpiVisionInfo = dstMxpivision->mutable_visioninfo();
        mxpiVisionInfo->set_format(1);
        mxpiVisionInfo->set_height(TENSOR_H);
        mxpiVisionInfo->set_width(TENSOR_W);
        mxpiVisionInfo->set_heightaligned(TENSOR_H);
        mxpiVisionInfo->set_widthaligned(TENSOR_W);
        // set MxpiVisionData by MemoryData
        MxTools::MxpiVisionData *mxpiVisionData = dstMxpivision->mutable_visiondata();
        mxpiVisionData->set_dataptr((uint64_t)newData.ptrData);
        mxpiVisionData->set_datasize(newData.size);
        mxpiVisionData->set_deviceid(newData.deviceId);
        mxpiVisionData->set_memtype((MxTools::MxpiMemoryType)newData.type);
        // visionlist->pair
        time_MxpiVisionList = std::make_pair(time_MxpiVisionList.first, mxpiVisionList);
        blockingMap_[id] = time_MxpiVisionList;
    }

    std::pair<std::chrono::high_resolution_clock::time_point, std::shared_ptr<MxTools::MxpiVisionList>> BlockingMap::Get(const uint32_t &id)
    {
        // add std::lock_guard
        std::lock_guard<std::mutex> guard(mtx_);
        if (blockingMap_.find(id) != blockingMap_.end())
        {
            return blockingMap_[id];
        }
        else
        {
            // If can't find the element, manually assign nullptr
            std::pair<std::chrono::high_resolution_clock::time_point, std::shared_ptr<MxTools::MxpiVisionList>> empty;
            using Time = std::chrono::high_resolution_clock;
            auto currentTime = Time::now();
            empty = std::make_pair(currentTime, nullptr);
            return empty;
        }
    }

    void BlockingMap::Clear(const uint32_t &id)
    {
        std::lock_guard<std::mutex> guard(mtx_);
        blockingMap_.erase(id);
        keys_.erase(id);
    }

    void BlockingMap::Reinsert(const uint32_t &id, std::shared_ptr<MxTools::MxpiVisionList> &mxpiVisionList)
    {
        // add std::lock_guard
        std::lock_guard<std::mutex> guard(mtx_);
        // get current timestamp
        using Time = std::chrono::high_resolution_clock;
        auto currentTime = Time::now();
        std::pair<std::chrono::high_resolution_clock::time_point,
                  std::shared_ptr<MxTools::MxpiVisionList>>
            time_MxpiVisionList;
        // set MxpiVisionInfo and MxpiVisionData
        time_MxpiVisionList = std::make_pair(currentTime, mxpiVisionList);
        blockingMap_[id] = time_MxpiVisionList;
        keys_.insert(id);
    }

    std::uint32_t BlockingMap::count(const uint32_t &id)
    {
        std::lock_guard<std::mutex> guard(mtx_);
        return blockingMap_.count(id);
    }

    size_t BlockingMap::Size() const
    {
        return blockingMap_.size();
    }

    std::vector<uint32_t> BlockingMap::Keys()
    {
        // id<->key
        std::vector<uint32_t> keys;
        std::lock_guard<std::mutex> guard(mtx_);
        for (auto iter = keys_.begin(); iter != keys_.end(); iter++)
        {
            keys.push_back(*iter);
        }
        return keys;
    }

    std::shared_ptr<MxTools::MxpiVisionList> BlockingMap::copyList(MxBase::MemoryData newData)
    {
        // create new shared_ptr MxpiVisionList;
        std::shared_ptr<MxTools::MxpiVisionList> dstMxpiVisionListSptr(new MxTools::MxpiVisionList,
                                                                       MxTools::g_deleteFuncMxpiVisionList);
        MxTools::MxpiVision *dstMxpivision = dstMxpiVisionListSptr->add_visionvec();
        MxTools::MxpiVisionInfo *mxpiVisionInfo = dstMxpivision->mutable_visioninfo();
        mxpiVisionInfo->set_format(1);
        mxpiVisionInfo->set_height(TENSOR_H);
        mxpiVisionInfo->set_width(TENSOR_W);
        mxpiVisionInfo->set_heightaligned(TENSOR_H);
        mxpiVisionInfo->set_widthaligned(TENSOR_W);
        // set MxpiVisionData by MemoryData
        MxTools::MxpiVisionData *mxpiVisionData = dstMxpivision->mutable_visiondata();
        mxpiVisionData->set_dataptr((uint64_t)newData.ptrData);
        mxpiVisionData->set_datasize(newData.size);
        mxpiVisionData->set_deviceid(newData.deviceId);
        mxpiVisionData->set_memtype((MxTools::MxpiMemoryType)newData.type);
        return dstMxpiVisionListSptr;
    }
}