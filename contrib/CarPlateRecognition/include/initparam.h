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

#ifndef Init_Param_H
#define Init_Param_H

// 结构体中定义了程序所需的所有参数
struct InitParam {
    
    // 系统参数
	bool checkTensor; // 判断是否检测输入至模型后处理的Tensor形状
	uint32_t deviceId; // 设备ID
    std::string DetecModelPath; // 车牌检测模型的存放路径
    std::string RecogModelPath; // 车牌识别模型的存放路径

    // 车牌检测模型的后处理参数
	float nmsThreshold; // 非极大值抑制的阈值
    float scoreThreshold; // 得分阈值，对生成的bbox进行阈值初筛
    int width ; // 车牌检测模型输入图像的宽
    int height; // 车牌检测模型输入图像的高
    std::vector<float> steps= {}; // 步长，用于生成特征图feature_map
    std::vector<std::vector<int>> min_sizes = {}; // 最小尺寸，用于生成锚框anchor
    std::vector<float> variances = {}; // 方差，用于对车牌检测模型的输出进行解码
    std::vector<int> scale = {}; // 尺度，用于对车牌检测模型的输出进行尺度还原

    // 车牌识别模型的后处理参数


};

#endif