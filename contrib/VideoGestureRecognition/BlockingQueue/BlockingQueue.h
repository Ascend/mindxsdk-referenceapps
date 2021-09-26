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
#ifndef VIDEOGESTUREREASONER_BLOCKINGQUEUE_H
#define VIDEOGESTUREREASONER_BLOCKINGQUEUE_H

#include <condition_variable>
#include <list>
#include <mutex>
#include <cstdint>
#include "MxBase/ErrorCode/ErrorCode.h"

template <typename T> class BlockingQueue {
public:
    explicit BlockingQueue(uint32_t maxSize = DEFAULT_MAX_QUEUE_SIZE) : maxSize(maxSize), isStopped(false) {}
    ~BlockingQueue() = default;

    APP_ERROR Pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex);

        while (queue_.empty() && !isStopped) {
            emptyCond.wait(lock);
        }

        if (isStopped) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.empty()) {
            return APP_ERR_QUEUE_EMPTY;
        } else {
            item = queue_.front();
            queue_.pop_front();
        }

        fullCond.notify_one();

        return APP_ERR_OK;
    }

    APP_ERROR Pop(T& item, uint32_t timeOutMs)
    {
        std::unique_lock<std::mutex> lock(mutex);
        auto realTime = std::chrono::milliseconds(timeOutMs);

        while (queue_.empty() && !isStopped) {
            emptyCond.wait_for(lock, realTime);
        }

        if (isStopped) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.empty()) {
            return APP_ERR_QUEUE_EMPTY;
        } else {
            item = queue_.front();
            queue_.pop_front();
        }

        fullCond.notify_one();

        return APP_ERR_OK;
    }

    APP_ERROR Push(const T& item, bool isWait = false)
    {
        std::unique_lock<std::mutex> lock(mutex);

        while (queue_.size() >= maxSize && isWait && !isStopped) {
            fullCond.wait(lock);
        }

        if (isStopped) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.size() >= maxSize) {
            return APP_ERR_QUEUE_FULL;
        }
        queue_.push_back(item);

        emptyCond.notify_one();

        return APP_ERR_OK;
    }

    APP_ERROR Push_Front(const T &item, bool isWait = false)
    {
        std::unique_lock<std::mutex> lock(mutex);

        while (queue_.size() >= maxSize && isWait && !isStopped) {
            fullCond.wait(lock);
        }

        if (isStopped) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.size() >= maxSize) {
            return APP_ERR_QUEUE_FULL;
        }

        queue_.push_front(item);

        emptyCond.notify_one();

        return APP_ERR_OK;
    }

    void Stop()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            isStopped = true;
        }

        fullCond.notify_all();
        emptyCond.notify_all();
    }

    void Restart()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            isStopped = false;
        }
    }

// if the queue is stopped ,need call this function to release the unprocessed items
    std::list<T> GetRemainItems()
    {
        std::unique_lock<std::mutex> lock(mutex);

        if (!isStopped) {
            return std::list<T>();
        }

        return queue_;
    }

    APP_ERROR GetBackItem(T &item)
    {
        if (isStopped) {
            return APP_ERR_QUEUE_STOPED;
        }

        if (queue_.empty()) {
            return APP_ERR_QUEUE_EMPTY;
        }

        item = queue_.back();
        return APP_ERR_OK;
    }

    APP_ERROR IsEmpty()
    {
        return queue_.empty();
    }

    APP_ERROR IsFull()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return queue_.size() >= maxSize;
    }

    int GetSize() const
    {
        return queue_.size();
    }

    std::mutex *GetLock()
    {
        return &mutex;
    }

    void Clear()
    {
        std::unique_lock<std::mutex> lock(mutex);
        queue_.clear();
    }

private:
    std::list<T> queue_;
    std::mutex mutex;
    std::condition_variable emptyCond;
    std::condition_variable fullCond;
    uint32_t maxSize;

    bool isStopped;

private:
    static const int DEFAULT_MAX_QUEUE_SIZE = 256;
};
#endif // MULTICHANNELVIDEODETECTION_BLOCKINGQUEUE_H
