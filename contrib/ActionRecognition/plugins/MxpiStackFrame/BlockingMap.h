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

#ifndef INC_FACE_BLOCKING_MAP_H
#define INC_FACE_BLOCKING_MAP_H

#include <map>
#include <set>
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <string>
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "MxBase/Log/Log.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/Proto/MxpiDataTypeDeleter.h"

/**
 * This is a thread safe map. All functions use std::lock_guard<std::mutex>.
*/

namespace MxPlugins {
    class BlockingMap {
    public:
        /**
         * @api
         * @brief Initialize a new mxpivisionlist for this id, and
         * insert the visiondata passed from upstream into the mxpivisionlist.
         * @param trackID, MemoryData
         */
        void Insert(const uint32_t &id, const MxBase::MemoryData newData);

        /**
         * @api
         * @brief Update mxpivisionlist and insert the visiondata for this id.
         * @param trackID, MemoryData
         */
        void Update(const uint32_t &id, const MxBase::MemoryData newData);

        /**
         * @api
         * @brief Reinsert mxpivisionlist and timestamp for this id.
         * @param trackID, MxpivisionList
         */
        void Reinsert(const uint32_t &id, std::shared_ptr<MxTools::MxpiVisionList> &mxpiVisionList);

        std::pair<std::chrono::high_resolution_clock::time_point, std::shared_ptr<MxTools::MxpiVisionList>>
        Get(const uint32_t &id);  // get std::pair instance
        void Clear(const uint32_t &id);  // erase map and id
        std::vector<uint32_t> Keys(); // save tackId into keys set
        size_t Size() const;

        std::uint32_t count(const uint32_t &id);   // return the number of elements of this id
        std::shared_ptr<MxTools::MxpiVisionList>
        copyList(const MxBase::MemoryData newData);     // set MxpiVisionInfo and MxpiVisionData
        BlockingMap();

        ~BlockingMap();

    private:
        std::mutex mtx_ = {};
        std::map<uint32_t, std::pair<std::chrono::high_resolution_clock::time_point,
                std::shared_ptr<MxTools::MxpiVisionList>>> blockingMap_ = {};
        std::set<uint32_t> keys_ = {};
    };
}
#endif

