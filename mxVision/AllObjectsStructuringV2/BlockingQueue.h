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

#include "MxBase/ErrorCode/ErrorCode.h"
#include <condition_variable>
#include <list>
#include <mutex>
#include <stdint.h>
#include <shared_mutex>

namespace MxBase
{
    const int DEFAULT_MAX_QUEUE_SIZE = 32;

    template <typename T>
    class BlockingQueue
    {
    public:
        explicit BlockingQueue(uint32_t maxSize = DEFAULT_MAX_QUEUE_SIZE) : maxSize_(maxSize) {}

        ~BlockingQueue() {}

        APP_ERROR Push(const T &item, bool isWait = false)
        {
            std::unique_lock<std::shared_mutex> lck(mutex_);
            while (queue_.size() >= maxSize_ && isWait)
            {
                fullCond_.wait(lck);
            }
            if (queue_.size() >= maxSize_)
            {
                return APP_ERR_QUEUE_FULL;
            }

            queue_.push_back(item);
            emptyCond_.notify_one();
            return APP_ERR_OK;
        }

        APP_ERROR Pop(T &item)
        {
            std::unique_lock<std::shared_mutex> lck(mutex_);
            if (queue_.empty())
            {
                return APP_ERR_QUEUE_EMPTY;
            }
            else
            {
                item = queue_.front();
                queue_.pop_front();
            }
            fullCond_.notify_one();
            return APP_ERR_OK;
        }

        APP_ERROR IsEmpty()
        {
            std::shared_lock<std::shared_mutex> lck(mutex_);
            return queue_.empty();
        }

        int GetSize()
        {
            std::shared_lock<std::shared_mutex> lck(mutex_);
            return queue_.size();
        }

    private:
        std::list<T> queue_;
        std::condition_variable_any fullCond_;
        std::condition_variable_any emptyCond_;
        std::shared_mutex mutex_;
        uint32_t maxSize_;
    };
}