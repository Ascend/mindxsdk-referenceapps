/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BLOCKING_MAP_H
#define BLOCKING_MAP_H

#include <map>
#include <mutex>
#include <set>

template<typename TKey, typename TValue> class BlockingMap {
public:
    BlockingMap() = default;

    ~BlockingMap() = default;

    void Insert(const TKey &id, TValue &streamData)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        blockingMap_[id] = streamData;
    }

    TValue Pop(const TKey &id)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (blockingMap_.find(id) != blockingMap_.end()) {
            auto streamData = blockingMap_[id];
            blockingMap_.erase(id);
            return streamData;
        } else {
            TValue value {};
            return value;
        }
    }

    size_t Size()
    {
        return blockingMap_.size();
    }

    bool Find(const TKey &id)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return blockingMap_.find(id) != blockingMap_.end();
    }

    typename std::map<TKey, TValue>::iterator RBegin()
    {
        auto iter = blockingMap_.end();
        iter--;
        return iter;
    }

    typename std::map<TKey, TValue>::iterator Begin()
    {
        return blockingMap_.begin();
    }

    typename std::map<TKey, TValue>::iterator End()
    {
        return blockingMap_.end();
    }

private:
    std::mutex mtx_ = {};
    std::map<TKey, TValue> blockingMap_ = {};
};

#endif
