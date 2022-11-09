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

const float pi = 3.1415926536f;

const int image_size[] = {512, 512};

const int line_size[] = {120, 1570}; // Line_height, Line_width

const int circle_center[] = {256, 256};

const int circle_radius = 250;

const int param_0 = 0;
const int param_1 = 1;
const int param_2 = 2;
const double param_0_5 = 0.5;
const double param_2_0 = 2.0;

void get_line_data(const vector<int64_t>& image,
    vector<unsigned int>* line_data) {
    float theta_col;
    int rho;
    float theta;
    int line_data_y;
    int line_data_x;
    int line_data_vec_index;
    int image_index;

    vector<unsigned int>& line_data_vec = (*line_data);

    for (int row_index = 0; row_index < line_size[0]; row_index++) {
        rho = circle_radius - row_index - param_1;
        for (int col_index = 0; col_index < line_size[1]; col_index++) {
            theta_col = col_index + param_1;
            line_data_vec_index = row_index * line_size[1] + col_index;

            theta = pi * param_2 / line_size[1] * theta_col;
            line_data_y = int(circle_center[0] + rho * cos(theta) + param_0_5);
            line_data_x = int(circle_center[1] - rho * sin(theta) + param_0_5);
            image_index = line_data_y * image_size[0] + line_data_x;
            line_data_vec[line_data_vec_index] = image[image_index];
        }
    }
    return;
}

void convert_1D_data(const vector<unsigned int>& line_data,
    vector<unsigned int>* scale_data,
    vector<unsigned int>* pointer_data) {
    vector<unsigned int>& scale_data_vec = (*scale_data);
    vector<unsigned int>& pointer_data_vec = (*pointer_data);
    for (int col_index = 0; col_index < line_size[1]; col_index++) {
        scale_data_vec[col_index] = 0;
        pointer_data_vec[col_index] = 0;
        for (int row_index = 0; row_index < line_size[0]; ++row_index) {
            int index = row_index * line_size[1] + col_index;
            pointer_data_vec[col_index] += (line_data[index] == param_1)?(param_1):(param_0);
            scale_data_vec[col_index] += (line_data[index] == param_2)?(param_1):(param_0);
        }
    }
    return;
}

void scale_mean_filt(const vector<unsigned int>& scale_data,
    vector<unsigned int>* scale_mean_data) {
    int sum = 0;
    int length = scale_data.size();
    for (int i = 0; i < length; i++) {
        sum = sum + scale_data[i];
    }
    float mean = float(sum) / length;

    vector<unsigned int>& scale_mean_data_vec = (*scale_mean_data);
    for (int i = 0; i < length; ++i) {
        if (float(scale_data[i]) >= mean) {
            scale_mean_data_vec[i] = scale_data[i];
        }
    }
    return;
}

void get_scale_location(const vector<unsigned int>& scale,
    vector<float>* scale_location) {
    LOCATION_SET scale_data; 
    scale_data.one_location = 0;
    scale_data.flag = 0;
    scale_data.start = 0;
    scale_data.end = 0;

    vector<float>& scale_location_vec = (*scale_location);
    for (int i = 0; i < line_size[1]; i++) {
        if ((scale[i] > 0 && scale[i + 1] > 0) && scale_data.flag == 0) {
            scale_data.start = i;
            scale_data.flag = param_1;
        }
        if ((scale[i] == 0 && scale[i + 1] == 0) && scale_data.flag == param_1) {
            scale_data.end = i - 1;
            scale_data.one_location = (scale_data.start + scale_data.end) / param_2_0;
            scale_location_vec.push_back(scale_data.one_location);
            scale_data.start = 0;
            scale_data.end = 0;
            scale_data.flag = 0;
        }
    }
}

void get_pointer_location(const vector<unsigned int>& pointer,
    float& pointer_location) {
    LOCATION_SET pointer_data; 
    pointer_data.one_location = 0;
    pointer_data.flag = 0;
    pointer_data.start = 0;
    pointer_data.end = 0;

    for (int i = 0; i < line_size[1]; i++) {
        if ((pointer[i] > 0 && pointer[i + 1] > 0) && pointer_data.flag == 0) {
                pointer_data.start = i;
                pointer_data.flag = param_1;
        }
        if ((pointer[i] == 0) && (pointer[i + 1] == 0) && pointer_data.flag == param_1) {
                pointer_data.end = i - param_1;
                pointer_data.one_location = (pointer_data.start + pointer_data.end) / param_2_0;
                pointer_data.start = 0;
                pointer_data.end = 0;
                pointer_data.flag = 0;
        }
    }
    pointer_location = pointer_data.one_location;
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
                float temp = (pointer_location - scale_location[i]) /
                    (scale_location[i + 1] - scale_location[i] + 1e-05);
                result->scales = i + 1 + temp;
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
        
        vector<unsigned int> line_data(line_size[1] * line_size[0], 0);
        vector<unsigned int> scale_data(line_size[1]);
        vector<unsigned int> pointer_data(line_size[1]);
        vector<unsigned int> scale_mean_data(line_size[1]);
        vector<float> scale_location;
        float pointer_location;
        READ_RESULT result;

        get_line_data(image[i_read], &line_data);
        convert_1D_data(line_data, &scale_data, &pointer_data);
        scale_mean_filt(scale_data, &scale_mean_data);
        get_scale_location(scale_mean_data, &scale_location);
        get_pointer_location(pointer_data, pointer_location);
        get_meter_reader(scale_location, pointer_location, &result);

        read_results_vec[i_read] = std::move(result);
    }
    return;
}
