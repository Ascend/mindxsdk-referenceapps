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

#include "PerformanceMonitor.h"
#include "MxBase/Log/Log.h"

#include <thread>

namespace AscendPerformanceMonitor {
/**
 * Init PerformanceMonitor
 * @param objects const reference to targets which need to record
 * @param enablePrint whether print performance message
 * @return status code of whether initialization is successful
 */
APP_ERROR PerformanceMonitor::Init(const std::vector<std::string> &objects, bool enablePrint)
{
    LogInfo << "PerformanceMonitor init start.";
    for (const std::string& object : objects) {
        std::vector<std::pair<int, double>> timeCostList;
        data.insert(std::pair<std::string, std::vector<std::pair<int, double>>>(object, timeCostList));
    }

    this->enablePrint = enablePrint;
    LogInfo << "PerformanceMonitor init success.";
    return APP_ERR_OK;
}

/**
 * De-init PerformanceMonitor
 * @return status code of whether de-initialization is successful
 */
APP_ERROR PerformanceMonitor::DeInit()
{
    LogInfo << "PerformanceMonitor deinit start.";
    std::unique_lock<std::mutex> lock(mutex);
    data.clear();

    LogInfo << "PerformanceMonitor deinit success.";
    return APP_ERR_OK;
}

/**
 * Collect target execute time
 * @param objectName const reference to target
 * @param timeCost curr time execute time
 * @return status code of whether collection si successful
 */
APP_ERROR PerformanceMonitor::Collect(const std::string &objectName, double timeCost)
{
    std::unique_lock<std::mutex> lock(mutex);

    currObject = objectName;
    auto objectInfo = data.find(currObject);
    if (objectInfo == data.end()) {
        LogError << "do not find " << currObject << " Collect.";
        return APP_ERR_COMM_FAILURE;
    }

    data[currObject].push_back(std::pair<int, double>(objectInfo->second.size(), timeCost));
    return APP_ERR_OK;
}

/**
 * Print x time performance statistics
 * @param currTime curr time index
 */
void PerformanceMonitor::Print(int currTime)
{
    if (!enablePrint) {
        return;
    }

    std::unique_lock<std::mutex> lock(mutex);

    LogInfo << currTime << " time performance statistics info: ";
    std::_Rb_tree_const_iterator<std::pair<const std::string, std::vector<std::pair<int, double>>>> iter;
    for (iter = data.begin(); iter != data.end(); iter++) {
        auto timeCostList = iter->second;
        auto total = timeCostList.size();
        double totalTimeCost = 0;
        for (auto timeCostPair : timeCostList) {
            totalTimeCost += timeCostPair.second;
        }
        auto average = total == 0 ? 0 : totalTimeCost / (double)total;
        LogInfo << iter->first << " total process frames: " << total << " average timecost: " << average << "ms.";
        data[iter->first].clear();
    }
}

/// ========== static Method ========== ///

/**
 * Print performance statistics
 * @param performanceMonitor const reference to the pointer to PerformanceMonitor
 * @param printInterval print interval
 */
void PerformanceMonitor::PrintStatistics(const std::shared_ptr<PerformanceMonitor> &performanceMonitor,
                                         int printInterval)
{
    int currTime = 0;
    while (true) {
        if (performanceMonitor->stopFlag) {
            LogInfo << "quit performance monitor.";
            break;
        }

        std::this_thread::sleep_for(std::chrono::seconds(printInterval));
        performanceMonitor->Print(++currTime);
    }
}
} // end AscendPerformanceMonitor