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
const int params_int[] = {0,1,2,3,4,5,6,7,8,9};

bool getLineData(const vector<int64_t>& image,
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
    return true;
}

bool convertToOneDimensionalData(const vector<unsigned int>& line_data,
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
    return true;
}

bool scaleAveFilt(const vector<unsigned int>& scale_data,
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
    return true;
}

bool getScaleLocation(const vector<unsigned int>& scale,
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
    return true;
}

bool getPointerLocation(const vector<unsigned int>& pointer,
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
    return true;
}


bool getMeterReader(const vector<float>& scale_location,
    float pointer_location, 
    READ_RESULT* result) {
    int scale_num = scale_location.size();
    READ_RESULT& result_struct = (*result);
    result_struct.scale_num = scale_num;
    result_struct.scales = -1;
    result_struct.ratio = -1;
    if (scale_num > 0) {
        for (int i = 0; i < scale_num - 1; i++) {
            bool scale_i_no_more_than_pointer =  (scale_location[i] <= pointer_location);
            bool pointer_less_than_scale_i1 = (pointer_location < scale_location[i + 1]);
            if ( scale_i_no_more_than_pointer && pointer_less_than_scale_i1) {
                float temp = (pointer_location - scale_location[i]) / (scale_location[i + 1] - scale_location[i] + 1e-05);
                result_struct.scales = i + 1 + temp;
            }
        }
        result_struct.ratio =(pointer_location - scale_location[0]) / (scale_location[scale_num - 1] - scale_location[0] + 1e-05);
    }
    return true;
}





void read_process(const vector<vector<int64_t>>& image,
    vector<READ_RESULT>* read_results,
    const int thread_num) {
    int read_num = image.size();
    vector<READ_RESULT>& read_results_vec = (*read_results);
    int process_flag_index = param_0;
    for (int img_read = 0; img_read < read_num; img_read++) {
        bool process_success_flag = false;
        vector<unsigned int> img_pointer_data(line_size[1]);
        vector<unsigned int> img_line_data(line_size[1] * line_size[0], 0);
        vector<float> img_scale_location;
        float img_pointer_location;
        vector<unsigned int> img_scale_data(line_size[1]);
        vector<unsigned int> img_scale_mean_data(line_size[1]);
        READ_RESULT img_result;
        if (process_flag_index == params_int[0]){
            process_success_flag = getLineData(image[img_read], &img_line_data);
            if (process_success_flag){
                process_flag_index++;
            }
        }
        if (process_flag_index == params_int[1]){
            process_success_flag = convertToOneDimensionalData(img_line_data, &img_scale_data, &img_pointer_data);
            if (process_success_flag){
                process_flag_index++;
            }
        }
        if (process_flag_index == params_int[2]){
            process_success_flag = scaleAveFilt(img_scale_data, &img_scale_mean_data);
            if (process_success_flag){
                process_flag_index++;
            }
        }
        if (process_flag_index == params_int[3]){
            process_success_flag = getScaleLocation(img_scale_mean_data, &img_scale_location);
            if (process_success_flag){
                process_success_flag = getPointerLocation(img_pointer_data, img_pointer_location);
                if (process_success_flag){
                    process_flag_index++;
                }
            } 
        }
        if (process_flag_index == params_int[4]){
            process_success_flag = getMeterReader(img_scale_location, img_pointer_location, &img_result);
            read_results_vec[img_read] = std::move(img_result);
            if (process_success_flag){
                process_flag_index=params_int[0];
            }
        }
        

        
        
        
        
        
    }
    return;
}
