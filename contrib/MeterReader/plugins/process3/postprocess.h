// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#define TYPE_THRESHOLD 40

#include <vector>


struct READ_RESULT {
    int scale_num;
    float scales;
    float ratio;
};

struct LOCATION_SET {
    float one_location;
    bool flag;
    unsigned int start;
    unsigned int end;
};


bool getLineData(const std::vector<int64_t>& image,
    std::vector<unsigned int>* line_data);

bool convertToOneDimensionalData(const std::vector<unsigned int>& line_data,
    std::vector<unsigned int>* scale_data,
    std::vector<unsigned int>* pointer_data);

bool scaleAveFilt(const std::vector<unsigned int>& scale_data,
    std::vector<unsigned int>* scale_mean_data);

bool getScaleLocation(const std::vector<unsigned int>& scale,
    std::vector<float>* scale_location);

bool getPointerLocation(const std::vector<unsigned int>& pointer,
    float& pointer_location);

bool getMeterReader(const std::vector<float>& scale_location,
    float pointer_location,
    READ_RESULT* result);

void read_process(const std::vector<std::vector<int64_t>>& image,
    std::vector<READ_RESULT>* read_results,
    const int thread_num);
