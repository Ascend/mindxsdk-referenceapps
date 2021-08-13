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

namespace AscendPerformanceMonitor {

APP_ERROR PerformanceMonitor::Init(const std::vector<std::string>& objects, bool enablePrint)
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

APP_ERROR PerformanceMonitor::DeInit()
{
    LogInfo << "PerformanceMonitor deinit start.";
    std::unique_lock<std::mutex> lock(mutex_);
    data.clear();

    LogInfo << "PerformanceMonitor deinit success.";
    return APP_ERR_OK;
}

APP_ERROR PerformanceMonitor::Process()
{
    return APP_ERR_OK;
}

APP_ERROR PerformanceMonitor::Collect(const std::string& objectName, double timeCost)
{
    std::unique_lock<std::mutex> lock(mutex_);

    currObject = objectName;
    auto objectInfo = data.find(currObject);
    if(objectInfo == data.end()) {
        LogError << "do not find " << currObject << " Collect.";
        return APP_ERR_COMM_FAILURE;
    }

    data[currObject].push_back(std::pair<int, double>(objectInfo->second.size(), timeCost));
    return APP_ERR_OK;
}

APP_ERROR PerformanceMonitor::Print(int currTime)
{
    if (!enablePrint) {
        return APP_ERR_OK;
    }

    std::unique_lock<std::mutex> lock(mutex_);

    LogInfo << currTime << " time performance statistics info: ";
    std::_Rb_tree_const_iterator<std::pair<const std::string, std::vector<std::pair<int, double>>>> iter;
    for (iter = data.begin(); iter != data.end(); iter++) {
        auto timeCostList = iter->second;
        auto total = timeCostList.size();
        double totalTimeCost = 0;
        for (auto timeCostPair : timeCostList) {
            totalTimeCost += timeCostPair.second;
        }
        auto average = total == 0 ? 0 : totalTimeCost / (double) total;
        LogInfo << iter->first << " total process: " << total << " average timecost: " << average << "ms.";
        data[iter->first].clear();
    }

    return APP_ERR_OK;
}

/// ========== static Method ========== ///
void PerformanceMonitor::PrintStatistics(const std::shared_ptr<PerformanceMonitor> &performanceMonitor,
                                         int printInterval)
{
    int currTime = 0;
    while(true) {
        if (performanceMonitor->stopFlag) {
            LogInfo << "quit performance monitor.";
            break;
        }

        sleep(printInterval);
        performanceMonitor->Print(++currTime);
    }
}
} // end AscendPerformanceMonitor