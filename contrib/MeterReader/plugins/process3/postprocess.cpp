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


#include <iostream>
#include <vector>
#include <utility>
#include <limits>
#include <cmath>
#include <chrono>  // NOLINT

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "postprocess.h"


using namespace std::chrono;  // NOLINT
using namespace std;

#define uint unsigned int

const float pi = 3.1415926536f;

const int image_size[] = {512, 512};

const int line_size[] = {120, 1570}; // Line_height, Line_width

const int circle_center[] = {256, 256};

const int circle_radius = 250;

const int param_1 = 1;
const int param_2 = 2;
const double param_0_5 = 0.5;
const double param_2_0 = 2.0;

void get_line_data(const vector<int64_t>& image,
    vector<uint>* line_data) {
    float theta;
    int rho;
    int line_data_x;
    int line_data_y;

    vector<uint>& line_data_vec = (*line_data);

    for (int row_index = 0; row_index < line_size[0]; row_index++) {
        for (int col_index = 0; col_index < line_size[1]; col_index++) {
            theta = pi * param_2 / line_size[1] * (col_index + param_1);
            rho = circle_radius - row_index - param_1;
            line_data_y = int(circle_center[0] + rho * cos(theta) + param_0_5);
            line_data_x = int(circle_center[1] - rho * sin(theta) + param_0_5);
            line_data_vec[row_index * line_size[1] + col_index] =
                image[line_data_y * image_size[0] + line_data_x];
        }
    }
    return;
}

void convert_1D_data(const vector<uint>& line_data,
    vector<uint>* scale_data,
    vector<uint>* pointer_data) {
    // Accumulte the number of positions whose label is 1 along the height axis.
    // Accumulte the number of positions whose label is 2 along the height axis.
    vector<uint>& scale_data_vec = (*scale_data);
    vector<uint>& pointer_data_vec = (*pointer_data);
    for (int col_index = 0; col_index < line_size[1]; col_index++) {
        scale_data_vec[col_index] = 0;
        pointer_data_vec[col_index] = 0;
        for (int row_index = 0; row_index < line_size[0]; ++row_index) {
            int index = row_index * line_size[1] + col_index;
            if (line_data[index] == 1) {
                ++pointer_data_vec[col_index];
            }
            else if (line_data[index] == param_2) {
                ++scale_data_vec[col_index];
            }
        }
    }
    return;
}

void scale_mean_filt(const vector<uint>& scale_data,
    vector<uint>* scale_mean_data) {
    int sum = 0;
    float mean = 0;
    int length = scale_data.size();
    for (int i = 0; i < length; i++) {
        sum = sum + scale_data[i];
    }
    mean = static_cast<float>(sum) / length;

    vector<uint>& scale_mean_data_vec = (*scale_mean_data);
    for (int i = 0; i < length; ++i) {
        if ((float)(scale_data[i]) >= mean) {
            scale_mean_data_vec[i] = scale_data[i];
        }
    }
    return;
}

void get_scale_location(const vector<uint>& scale,
    vector<float>* scale_location) {
    float one_scale_location = 0;
    bool scale_flag = 0;
    uint scale_start = 0;
    uint scale_end = 0;

    vector<float>& scale_location_vec = (*scale_location);
    for (int i = 0; i < line_size[1]; i++) {
        if (scale[i] > 0 && scale[i + 1] > 0) {
            if (scale_flag == 0) {
                scale_start = i;
                scale_flag = param_1;
            }
        }
        if (scale_flag == param_1) {
            if (scale[i] == 0 && scale[i + 1] == 0) {
                scale_end = i - 1;
                one_scale_location = (scale_start + scale_end) / param_2_0;
                scale_location_vec.push_back(one_scale_location);
                scale_start = 0;
                scale_end = 0;
                scale_flag = 0;
            }
        }
    }
}

void get_pointer_location(const vector<uint>& pointer,
    float& pointer_location) {
    pointer_location = 0;
    bool pointer_flag = 0;
    uint pointer_start = 0;
    uint pointer_end = 0;
    for (int i = 0; i < line_size[1]; i++) {
        if (pointer[i] > 0 && pointer[i + 1] > 0) {
            if (pointer_flag == 0) {
                pointer_start = i;
                pointer_flag = param_1;
            }
        }
        if (pointer_flag == param_1) {
            if ((pointer[i] == 0) && (pointer[i + 1] == 0)) {
                pointer_end = i - param_1;
                pointer_location = (pointer_start + pointer_end) / param_2_0;
                pointer_start = 0;
                pointer_end = 0;
                pointer_flag = 0;
            }
        }
    }
}


void get_meter_reader(const vector<float>& scale_location,
    float pointer_location, READ_RESULT* result) {
    int scale_num = scale_location.size();
    result->scale_num = scale_num;
    result->scales = -1;
    result->ratio = -1;
    if (scale_num > 0) {
        for (int i = 0; i < scale_num - 1; i++) {
            if (scale_location[i] <= pointer_location &&
                pointer_location < scale_location[i + 1]) {
                result->scales = i + 1 +
                    (pointer_location - scale_location[i]) /
                    (scale_location[i + 1] - scale_location[i] + 1e-05);
            }
        }
        result->ratio =
            (pointer_location - scale_location[0]) /
            (scale_location[scale_num - 1] - scale_location[0] + 1e-05);
    }
    return;
}


void read_process(const vector<vector<int64_t>>& image,
    vector<READ_RESULT>* read_results,
    const int thread_num) {
    int read_num = image.size();
    vector<READ_RESULT>& read_results_vec = (*read_results);
#pragma omp parallel for num_threads(thread_num)
    for (int i_read = 0; i_read < read_num; i_read++) {

        vector<uint> line_data(line_size[1] * line_size[0], 0);
        get_line_data(image[i_read], &line_data);


        vector<uint> scale_data(line_size[1]);
        vector<uint> pointer_data(line_size[1]);
        convert_1D_data(line_data, &scale_data, &pointer_data);


        vector<uint> scale_mean_data(line_size[1]);
        scale_mean_filt(scale_data, &scale_mean_data);


        vector<float> scale_location;
        get_scale_location(scale_mean_data, &scale_location);

        float pointer_location;
        get_pointer_location(pointer_data, pointer_location);

        READ_RESULT result;
        get_meter_reader(scale_location, pointer_location, &result);

        read_results_vec[i_read] = std::move(result);
    }
    return;
}
