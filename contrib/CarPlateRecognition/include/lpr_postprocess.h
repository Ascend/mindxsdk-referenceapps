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

#ifndef LPR_POSTPROCESS_H
#define LPR_POSTPROCESS_H
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "initparam.h"


// 车牌识别模型的后处理类
class LPRPostProcess
{
public:
    LPRPostProcess() = default; // 构造函数
    ~LPRPostProcess() = default; // 析构函数
    APP_ERROR init(const InitParam &initParam); // 后处理初始化函数
    APP_ERROR deinit(); // 后处理解初始化函数
    APP_ERROR process(std::vector<MxBase::TensorBase> recog_outputs, MxBase::ObjectInfo& objectInfo); // 后处理主流程函数
protected:

private:

};

#endif