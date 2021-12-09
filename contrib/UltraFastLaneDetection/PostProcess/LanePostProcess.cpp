/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "LanePostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace {
const int SCALE = 32;
const int YOLO_INFO_DIM = 5;
auto uint8Deleter = [] (uint8_t* p) { };
const int DATA_SET1 = 201;   // 数据组数
const int DATA_SET2 = 200;
const int DOT = 18;       // 车道线上的点
const int LANE = 4;       // 车道线线数
float prob[DATA_SET2][DOT][LANE];
float out_k[DATA_SET2][DOT][LANE];
}
namespace MxBase {
    APP_ERROR LanePostProcess::Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) {
        LogDebug << "Start to Init LanePostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        LogDebug << "End to Init LanePostProcess.";
        return APP_ERR_OK;
    }
    APP_ERROR LanePostProcess::DeInit() {
        return APP_ERR_OK;
    }
    void LanePostProcess::my_softmax(int x,int y)
    {
        float sum = 0.0;
        for (int i = 0; i < DATA_SET2; ++i){
            sum += exp(out_k[i][x][y]);
        }
        for (int i = 0; i < DATA_SET2; ++i){
            prob[i][x][y] = exp(out_k[i][x][y]) / sum;
        }
    }
    void LanePostProcess::ObjectDetectionOutput(const std::vector <TensorBase> &tensors,
                                                std::vector <std::vector<ObjectInfo>> &objectInfos,
                                                const std::vector <ResizedImageInfo> &resizedImageInfos) {
        LogDebug << "LanePostProcess start to write results.";
        if (tensors.size() == 0) {
            return;
        }
        auto shape = tensors[0].GetShape();
        if (shape.size() == 0) {
            return;
        }
        uint32_t batchSize = shape[0];
        for (uint32_t i = 0; i < batchSize; i++) {
            std::vector <std::shared_ptr<void>> featLayerData = {};
            std::vector <std::vector<size_t>> featLayerShapes = {};
            for (uint32_t j = 0; j < tensors.size(); j++) {
                auto dataPtr = (uint8_t *) GetBuffer(tensors[j], i);
                std::shared_ptr<void> tmpPointer;
                tmpPointer.reset(dataPtr, uint8Deleter);
                featLayerData.push_back(tmpPointer);
                shape = tensors[j].GetShape();
                std::vector <size_t> featLayerShape = {};
                for (auto s : shape) {
                    featLayerShape.push_back((size_t) s);
                }
                featLayerShapes.push_back(featLayerShape);
            }
            std::vector <ObjectInfo> objectInfo;
            int netindex = 0;
            std::shared_ptr<void> netout01 = featLayerData[0];
            float out[DATA_SET1][DOT][LANE];
            float out_j1[DATA_SET1][DOT][LANE];
            float out_m[DATA_SET2][DOT][LANE];
            float loc[DOT][LANE];
            float out_j2[DOT][LANE];
            float out_j3[DOT][LANE];
            for (int i = 0; i < DATA_SET1; ++i) {
                for(int j = 0; j < DOT; ++j){
                    for(int k = 0; k < LANE; ++k){
                        out[i][j][k] = static_cast<float *>(netout01.get())[netindex];
                        netindex = netindex + 1;
                    }
                }
            }
            for(int i = 0; i < DATA_SET1; i++){
                for(int j = 0; j < DOT; j ++){
                     for(int k = 0; k < LANE; k++){
                        out_j1[i][j][k] = out[i][17-j][k];
                     }
                }
            }
            for(int i = 0; i < DATA_SET2; i++){
                for(int j = 0; j < DOT; j++){
                    for(int k = 0;k < LANE; k++){
                        out_k[i][j][k] = out_j1[i][j][k];
                    }
                }
            }
            for(int j = 0; j < DOT; j++){
                for(int k = 0; k < LANE; k++){
                    my_softmax(j,k);
                }
            }
            for(int i = 0; i < DATA_SET2; i++){
                for(int j = 0; j < DOT; j++){
                    for(int k = 0; k < LANE; k++){
                        out_m[i][j][k] = prob[i][j][k]*(i+1);
                    }
                }
            }
            for(int k = 0; k < LANE; k++){
                for(int j = 0; j < DOT; j++){
                    float sum_1 = 0.0;
                    for(int i = 0; i < DATA_SET2; i++){
                        sum_1 = out_m[i][j][k] + sum_1;
                    }
                    loc[j][k] = sum_1;
                }
            }
            for(int j = 0; j < DOT; j++){
                for(int k = 0; k < LANE; k++){
                float max = -10000.0;
                    for(int i = 0; i < DATA_SET1; i++){
                        out_j1[i][j][k] = out[i][17-j][k];
                        if(out_j1[i][j][k] > max)
                        {
                            max = out_j1[i][j][k];
                            out_j2[j][k] = i;
                        }
                    }
                }
            }
            for(int j = 0; j < DOT; j++){
                for(int k = 0; k < LANE; k++){
                    if(out_j2[j][k] == DATA_SET2){
                        out_j3[j][k] == 0;
                    }
                    else{
                        out_j3[j][k] = loc[j][k];
                    }
                }
            }
            for(int i = 0; i < DOT; ++i){
                MxBase::ObjectInfo det;
                det.x0 = out_j3[i][0];
                det.y0 = out_j3[i][1];
                det.x1 = out_j3[i][2];
                det.y1 = out_j3[i][3];
                objectInfo.push_back(det);
            }
            objectInfos.push_back(objectInfo);
        }
        LogDebug << "LanePostProcess write results successed.";
    }

    APP_ERROR LanePostProcess::Process(const std::vector <TensorBase> &tensors,
                                       std::vector <std::vector<ObjectInfo>> &objectInfos,
                                       const std::vector <ResizedImageInfo> &resizedImageInfos,
                                       const std::map <std::string, std::shared_ptr<void>> &paramMap) {
        LogDebug << "Start to Process LanePostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        if (resizedImageInfos.size() == 0) {
            ret = APP_ERR_INPUT_NOT_MATCH;
            LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary for LanePostProcess.";
            return ret;
        }
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "CheckAndMoveTensors failed.";
            return ret;
        }

        ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
        LogObjectInfos(objectInfos);
        LogDebug << "End to Process LanePostProcess.";
        return APP_ERR_OK;
    }

/*
 * @description: Compare the confidences between 2 classes and get the larger one
 */

    extern "C" {
    std::shared_ptr <MxBase::LanePostProcess> GetObjectInstance() {
        LogInfo << "Begin to get LanePostProcess instance.";
        auto instance = std::make_shared<MxBase::LanePostProcess>();
        LogInfo << "End to get LanePostProcess instance.";
        return instance;
    }
    }
}