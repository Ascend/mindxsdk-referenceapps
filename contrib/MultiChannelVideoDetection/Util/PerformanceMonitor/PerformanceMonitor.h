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

#ifndef MULTICHANNELVIDEODETECTION_PERFORMANCEMONITOR_H
#define MULTICHANNELVIDEODETECTION_PERFORMANCEMONITOR_H

#include <map>
#include <vector>
#include <mutex>
#include <memory>
#include "MxBase/ErrorCode/ErrorCode.h"

namespace AscendPerformanceMonitor {
// PerformanceMonitor config
const uint32_t DEFAULT_PRINT_INTERVAL = 1;

class PerformanceMonitor {
public:
    PerformanceMonitor() = default;
    ~PerformanceMonitor() = default;
    APP_ERROR Init(const std::vector<std::string> &objects, bool enablePrint);
    APP_ERROR DeInit();

    APP_ERROR Collect(const std::string &objectName, double timeCost);
    void Print(int currTime);

public:
    static void PrintStatistics(const std::shared_ptr<PerformanceMonitor> &performanceMonitor, int printInterval);

public:
    // running flag
    bool stopFlag = false;

private:
    std::map<std::string, std::vector<std::pair<int, double>>> data;

    std::mutex mutex;

    std::string currObject;

    // whether print performance message
    bool enablePrint = true;
};
} // end AscendPerformanceMonitor
#endif // MULTICHANNELVIDEODETECTION_PERFORMANCEMONITOR_H
