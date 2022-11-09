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
using namespace std;


struct READ_RESULT {
    int scale_num;
    float scales;
    float ratio;
};

struct LOCATION_SET{
    float one_location;
    bool flag;
    unsigned int start;
    unsigned int end;
};


void get_line_data(const vector<int64_t>& image,
    vector<unsigned int>* line_data);

void convert_1D_data(const vector<unsigned int>& line_data,
    vector<unsigned int>* scale_data,
    vector<unsigned int>* pointer_data);

void scale_mean_filt(const vector<unsigned int>& scale_data,
    vector<unsigned int>* scale_mean_data);

void get_scale_location(const vector<unsigned int>& scale,
    vector<float>* scale_location);

void get_pointer_location(const vector<unsigned int>& pointer,
    float& pointer_location);

void get_meter_reader(const vector<float>& scale_location,
    float pointer_location,
    READ_RESULT* result);


void read_process(const vector<vector<int64_t>>& image,
    vector<READ_RESULT>* read_results,
    const int thread_num);
