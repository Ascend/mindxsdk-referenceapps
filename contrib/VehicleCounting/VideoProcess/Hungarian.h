/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef STREAM_PULL_SAMPLE_HUNGARIAN_H
#define STREAM_PULL_SAMPLE_HUNGARIAN_H

#include <vector>
#include <memory>
#include "DataType.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/ErrorCode/ErrorCodes.h"
//#include "acl/acl.h"

struct g_hungarianhandle {
    int rows;
    int cols;
    int max;
    int *resX;
    int *resY;
    bool transpose;
    std::shared_ptr<int> adjMat;
    std::shared_ptr<int> xMatch;
    std::shared_ptr<int> yMatch;
    std::shared_ptr<int> xValue;
    std::shared_ptr<int> yValue;
    std::shared_ptr<int> slack;
    std::shared_ptr<int> xVisit;
    std::shared_ptr<int> yVisit;
};

APP_ERROR hungarianhandleinit(hungarianhandle &handle, int row, int cols);
int hungariansolve(hungarianhandle &handle, const std::vector<std::vector<int>> &gcost, int rows, int cols);

#endif //STREAM_PULL_SAMPLE_HUNGARIAN_H
