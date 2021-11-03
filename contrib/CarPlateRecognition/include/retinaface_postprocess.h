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

#ifndef RetinaFace_PsotProcess_H
#define RetinaFace_PsotProcess_H
#include "opencv2/opencv.hpp"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "initparam.h"


// 锚框anchor结构体变量
struct box{
    float cx; // anchor中心点(central)的x坐标
    float cy; // anchor中心点的y坐标
    float sx; // x轴方向的步幅(step)
    float sy; // y轴方向的步幅
};


// 车牌检测模型RetinaFace的后处理类
class RetinaFace_PostProcess : public MxBase::ObjectPostProcessBase
{
public:
    RetinaFace_PostProcess() = default; // 构造函数
    ~RetinaFace_PostProcess()= default; // 析构函数
    APP_ERROR Init(const InitParam &initParam); // 后处理初始化函数
    APP_ERROR DeInit(); // 后处理解初始化函数
    APP_ERROR Process(std::vector<MxBase::TensorBase> detect_outputs, std::vector<MxBase::ObjectInfo>& objectInfos, const MxBase::ResizedImageInfo resizedImageInfo); // 后处理主流程函数

protected:
    void nms(std::vector<MxBase::ObjectInfo> &input_boxes, float NMS_THRESH); // 极大值抑制函数
    void GenerateAnchor(std::vector<box> &anchor, int w, int h); // 锚框生成函数
    void SetDefaultParams(); // 将后处理所需的参数设置为默认值
    static inline bool cmp(MxBase::ObjectInfo a, MxBase::ObjectInfo b); // 比较两个ObjectInfo类型变量的置信度大小

private:
    float nmsThreshold_; // 非极大值抑制的阈值
    float scoreThreshold_; // 得分阈值，对生成的bbox进行阈值初筛
    int width_ ; // 模型输入图像的宽(经resize后)
    int height_; // 模型输入图像的高(经resize后)
    std::vector<float> steps_= {}; // 步长，用于生成特征图feature_map
    std::vector<std::vector<int>> min_sizes_ = {}; // 最小尺寸，用于生成锚框anchor
    std::vector<float> variances_ = {}; // 方差，用于对模型的输出进行解码
    std::vector<int> scale_ = {}; // 尺度，用于对模型的输出进行尺度还原
};
#endif 
