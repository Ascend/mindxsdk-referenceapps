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

#include "RTM3DPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"


// 在不同的名字空间下可以存在同名的函数、类、变量，它们互不影响    using namespace std方便调用cout、endl等所以不容易发现其实这个习惯可能会出现问题
namespace {
const float SCORE_THRESH = 0.39;
const int TOPK_CANDIDATES = 100;
const int TENSOR_SIZE = 4;
const int NETOUT01_DIM1 = 1;
const int NETOUT01_DIM2 = 3;
const int NETOUT01_DIM3 = 104;   // 四个tensor的第三维度和第四维度是相同的，之后其他netout0*通用
const int NETOUT01_DIM4 = 320;   // 四个tensor的第三维度和第四维度是相同的，之后其他netout0*通用
const int PAD = 1;
const int NETOUT02_DIM2 = 16;
const int NETOUT03_DIM2 = 2;
const int FRAME_POINT = 8;

auto uint8Deleter = [] (uint8_t* p) { };
}

// 在不同的名字空间下可以存在同名的函数、类、变量，它们互不影响    using namespace std方便调用cout、endl等所以不容易发现其实这个习惯可能会出现问题
namespace MxBase {
    APP_ERROR RTM3DPostProcess::Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) {
        LogDebug << "Start to Init RTM3DPostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        return APP_ERR_OK;
    }

    APP_ERROR RTM3DPostProcess::DeInit() {
        return APP_ERR_OK;
    }

    bool RTM3DPostProcess::IsValidTensors(const std::vector <TensorBase> &tensors) const {
        long tensor_size[4] = {66560,66560,532480,99840};   // 四个tensor各包含多少数
        if (tensors.size() != (size_t) RTM3DType_) {   // 强制类型转换，int转换成size_t
            LogError << "number of tensors (" << tensors.size() << ") " << "is unequal to RTM3DType_("
                     << RTM3DType_ << ")";
            return false;
        }
        if (RTM3DUsing_) {
            std::cout << "-------tensors.size() is" << tensors.size() << std::endl;
            for (size_t i = 0; i < tensors.size(); i++) {
                auto shape = tensors[i].GetShape();   // shape是一个数组，含四个数
                if (shape.size() < 4) {
                    LogError << "dimensions of tensor [" << i << "] is less than 4.";
                    return false;
                }
                long sumNumber = 1;
                int startIndex = 0;
                int endIndex = 4;
                std::cout << "--------output tensor[" << i << "] is" << i << std::endl;
                for (int i = startIndex; i < endIndex; i++) {
                    std::cout << "--------shape[" << i << "] is" << shape[i] << std::endl;
                    sumNumber *= shape[i];    // 每个shape中包含多少数
                }
                if (sumNumber != tensor_size[i]) {
                    LogError << "sumNumber[" << i << "] is not true";
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    void downdata(std::shared_ptr<void> &netout01,
                  std::shared_ptr<void> &netout02,
                  std::shared_ptr<void> &netout03,
                  std::shared_ptr<void> &netout04,
                  std::vector <std::shared_ptr<void>> featLayerData){
        for (int i = 0; i < TENSOR_SIZE; ++i) {
            switch(i){
                case 0:
                    netout04 = featLayerData[i];  // 对应模型输出的1 * 2 * 104 * 320
                    break;
                case 1:
                    netout03 = featLayerData[i];  // 对应模型输出的1 * 2 * 104 * 20
                    break;
                case 2:
                    netout02 = featLayerData[i];  // 对应模型输出的1 * 16 * 104 * 320
                    break;
                case 3:
                    netout01 = featLayerData[i];  // 对应模型输出的1 * 3 * 104 * 320
                    break;
                default:
                    std::cout << "data error" << std::endl;
            }
        }
    }

    void MaxPool(float padbegindata[NETOUT01_DIM2][PAD + NETOUT01_DIM3 + PAD][PAD + NETOUT01_DIM4 + PAD],
                 float begindata[NETOUT01_DIM2][NETOUT01_DIM3][NETOUT01_DIM4],
                 float pooldata[NETOUT01_DIM2][NETOUT01_DIM3][NETOUT01_DIM4],
                 int netout01_dim2,
                 std::shared_ptr<void> netout01){
        static int netindex01 = 0;
        static int netindex02 = 0;
        for(int j = 0; j < (PAD + NETOUT01_DIM3 + PAD); ++j){
        // netout01中的第三维度 + pad * 2   104 + 1 * 2
            for(int k = 0; k < (PAD + NETOUT01_DIM4 + PAD); ++k){
             // netout01中的第四维度 + pad * 2   320 + 1 * 2
                if((j == 0) || (j == (PAD + NETOUT01_DIM3 + PAD - 1)) || (k == 0) || (k == (PAD + NETOUT01_DIM4 + PAD - 1))){
                 // 在池化前补边，补充的数据都置0， pad=1
                    padbegindata[netout01_dim2][j][k] = 0;
                }
                else{
                    padbegindata[netout01_dim2][j][k] = fastmath::sigmoid(static_cast<float *>(netout01.get())[netindex01]);
                        // 将第一个tensor从一维变成三维，并且进行了sigmoid即f(x) = 1. / (1 + np.exp(-x))
                    netindex01 = netindex01 + 1;
                }
            }
        }

        float key[9];                      // 卷积核3 * 3 = 9
        for(int j = 0; j < NETOUT01_DIM3; ++j){      // netout01中的第三维度  104  步长为1
            for(int k = 0; k < NETOUT01_DIM4; ++k){  // netout01中的第四维度  320
                key[0] = padbegindata[netout01_dim2][j][k];     // 卷积核为3*3
                key[1] = padbegindata[netout01_dim2][j][k + 1];
                key[2] = padbegindata[netout01_dim2][j][k + 2];
                key[3] = padbegindata[netout01_dim2][j + 1][k];
                key[4] = padbegindata[netout01_dim2][j + 1][k + 1];
                key[5] = padbegindata[netout01_dim2][j + 1][k + 2];
                key[6] = padbegindata[netout01_dim2][j + 2][k];
                key[7] = padbegindata[netout01_dim2][j + 2][k + 1];
                key[8] = padbegindata[netout01_dim2][j + 2][k + 2];
                int max_idx = std::max_element(key, key + 9) - key;  // 找到九个值中最大的下标
                pooldata[netout01_dim2][j][k] = key[max_idx];       // 将最大值赋给接受池化后的数据的数组pooldata
                begindata[netout01_dim2][j][k] = fastmath::sigmoid(static_cast<float *>(netout01.get())[netindex02]);
                    // 将第一个tensor从一维变成三维，并且进行了sigmoid即f(x) = 1. / (1 + np.exp(-x))
                if(begindata[netout01_dim2][j][k] != pooldata[netout01_dim2][j][k]){   // 相当于找出极大值
                    begindata[netout01_dim2][j][k] = 0;
                }
                netindex02 = netindex02 +1;
            }
        }
    }

    void ObtainMainProj2d(float begindata[NETOUT01_DIM2][NETOUT01_DIM3][NETOUT01_DIM4],
                          float topdata[],
                          int indices[],
                          float scores[],
                          std::vector <int> &xdata,
                          std::vector <int> &ydata,
                          std::vector <int> &clses,
                          std::vector <float> &m_scores){
        int netindex = 0;
        for(int j = 0; j < (NETOUT01_DIM2 * NETOUT01_DIM3); ++j){   // netout01中的第二、三维度 3 104
            for(int k = 0; k < NETOUT01_DIM4; ++k){   // netout01中的第四维度  320
                topdata[netindex] = begindata[j / NETOUT01_DIM3][j % NETOUT01_DIM3][k];   // 将处理好的数据从三维变成一维
                netindex = netindex + 1;
            }
        }

        for(int i = 0; i < TOPK_CANDIDATES; ++i){
            int max_idx = std::max_element(topdata, topdata + (NETOUT01_DIM2 * NETOUT01_DIM3 * NETOUT01_DIM4)) - topdata;
            // 在3 * 104 * 320个数中找到topdata数组里面的最大值下标
            indices[i] = max_idx;
            scores[i] = topdata[max_idx];
            topdata[max_idx] = 0;          // 最大值置0
        }

        for(int i = 0; i < TOPK_CANDIDATES; ++i){
            if(scores[i] > SCORE_THRESH){  // 原本项目是0.4，但是转移到服务器上数据有一点偏差，阈值可能不太合适
                int cls = indices[i] / (NETOUT01_DIM3 * NETOUT01_DIM4);   // 104 * 320 = 33280
                int xy = indices[i] % (NETOUT01_DIM3 * NETOUT01_DIM4);    // 104 * 320 = 33280
                int x = xy % NETOUT01_DIM4;    // 得到列
                int y = xy / NETOUT01_DIM4;    // 得到行
                xdata.push_back(x);
                ydata.push_back(y);
                clses.push_back(cls);
                m_scores.push_back(scores[i]);
            }
        }
    }

    void DataFromSameIndex(std::vector <std::vector <float>> &lay2data,
                           std::vector <std::vector <float>> &lay3data,
                           std::shared_ptr<void> netout02,
                           std::shared_ptr<void> netout03,
                           int size,
                           std::vector <int> xdata,
                           std::vector <int> ydata){
        for(int i = 0; i < size; ++i){  // 在netout02中找到x==第三维中索引和y==第四维中索引的值
            std::vector <float> lay2dim = {};  // 相当于把前一次的vector清掉了
            for(int j = 0; j < NETOUT02_DIM2; ++j){    // netout02中的第二维度 16
                lay2dim.push_back(static_cast<float *>(netout02.get())[(NETOUT01_DIM3 * NETOUT01_DIM4) * j   // 104 * 320 = 33280
                                                                       + ydata[i] * NETOUT01_DIM4 + xdata[i]]); 
                // 从第一个tensor数据获得的下标找到第二个tensor中对应下标的值
            }
            lay2data.push_back(lay2dim);   // 得到16 * xdata.size() 的向量
        }

        for(int i = 0; i < size; ++i){ // 在netout03中找到x==第三维中索引和y==第四维中索引的值
            std::vector <float> lay3dim = {};  // 相当于把前一次的vector清掉了
            for(int j = 0; j < NETOUT03_DIM2; ++j){        // netout03中的第二维度 2
                lay3dim.push_back(fastmath::sigmoid(
                    static_cast<float *>(netout03.get())[33280 * j + ydata[i] * 320 + xdata[i]]));    // 104 * 320 = 33280
                // 从第一个tensor数据获得的下标找到第三个tensor中对应下标的值，并且进行sigmoid即f(x) = 1. / (1 + np.exp(-x))
            }
            lay3data.push_back(lay3dim);
        }
    }

    void FrameCoordinates(int size,
                          std::vector <std::vector <std::vector <float>>> &v_projs_regress,
                          std::vector <std::vector <float>> lay2data,
                          std::vector <float> &xdataf,
                          std::vector <float> &ydataf,
                          std::vector <int> xdata,
                          std::vector <int> ydata,
                          std::vector <std::vector <float>> lay3data){
        for(int i = 0; i < size; ++i){
        // 将 16 * xdata.size() 的向量lay2data转成 xdata.size()个  8 * 2 的三维向量v_projs_regress
        // 原理是将16 * xdata.size()的向量先转成8个2 * xdata.size()的三维向量，将得到的三维向量转化成xdata.size()个8 * 2 的三维向量
            std::vector <std::vector <float>> out2dim = {};
            for(int j = 0; j < NETOUT02_DIM2; ++j){
                std::vector <float> outhang = {};
                int k = 0;
                outhang.push_back(lay2data[i][j + (k++)]);
                outhang.push_back(lay2data[i][j + (k++)]);
                out2dim.push_back(outhang);
                j = j + 1;
            }
            v_projs_regress.push_back(out2dim);
        }

        for(int i = 0; i < size; ++i){
            xdataf.push_back(xdata[i] + lay3data[i][0]);
            ydataf.push_back(ydata[i] + lay3data[i][1]);
        }

        for(int i = 0; i < size; ++i){
            for(int j = 0; j < FRAME_POINT; ++j){
                v_projs_regress[i][j][0] = v_projs_regress[i][j][0] + xdataf[i];
                v_projs_regress[i][j][0] = v_projs_regress[i][j][0] * 4;
                v_projs_regress[i][j][1] = v_projs_regress[i][j][1] + ydataf[i];
                v_projs_regress[i][j][1] = v_projs_regress[i][j][1] * 4;
            }
        }
    }

    void RTM3DPostProcess::ObjectDetectionOutput(const std::vector <TensorBase> &tensors,
                                                 std::vector <std::vector<ObjectInfo>> &objectInfos) {
        LogDebug << "RTM3DPostProcess start to write results.";
        if (tensors.size() == 0) {
            return;
        }
        auto shape = tensors[0].GetShape();  // 1 2 104 320??
        if (shape.size() == 0) {
            return;
        }
        uint32_t batchSize = shape[0];
        std::cout << "--------shape[0] is" << shape[0] << std::endl;
        for (uint32_t i = 0; i < batchSize; i++) {    // 就一次
            std::vector <std::shared_ptr<void>> featLayerData = {}; // 放om的输出数据
            std::vector <std::vector<size_t>> featLayerShapes = {}; // 放om的输出格式
            for (uint32_t j = 0; j < tensors.size(); j++) {  // 循环4次  4个输出   我应该得倒过来
                auto dataPtr = (uint8_t *) GetBuffer(tensors[j], i);  // 获取tensor[0]/[1]/[2]/[3]里面数据的指针
                std::shared_ptr<void> tmpPointer;
                tmpPointer.reset(dataPtr, uint8Deleter); // 首先生成新对象，然后引用计数减1，引用计数为0，故析构tmpPointer
                                                         // 最后将新对象dataPtr的指针交给智能指针
                featLayerData.push_back(tmpPointer);
                shape = tensors[j].GetShape();
                std::vector <size_t> featLayerShape = {};
                for (auto s : shape) {
                    featLayerShape.push_back((size_t) s);
                }
                featLayerShapes.push_back(featLayerShape);
            }
            std::vector <ObjectInfo> objectInfo;
            GenerateBbox(objectInfo, featLayerData, featLayerShapes);
            objectInfos.push_back(objectInfo);
        }
        LogDebug << "RTM3DPostProcess write results successed.";
    }

    APP_ERROR RTM3DPostProcess::Process(const std::vector <TensorBase> &tensors,
                                        std::vector <std::vector<ObjectInfo>> &objectInfos,
                                        const std::vector <ResizedImageInfo> &resizedImageInfos,
                                        const std::map <std::string, std::shared_ptr<void>> &paramMap) {
        LogDebug << "Start to Process RTM3DPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "CheckAndMoveTensors failed.";
            return ret;
        }

        ObjectDetectionOutput(inputs, objectInfos);
        LogObjectInfos(objectInfos);    // 输出自己需要的信息，在main.py里面能取出来
        LogDebug << "End to Process RTM3DPostProcess.";
        return APP_ERR_OK;
    }

    void RTM3DPostProcess::GenerateBbox(std::vector <MxBase::ObjectInfo> &detBoxes,
                                        std::vector <std::shared_ptr<void>> featLayerData,
                                        const std::vector <std::vector<size_t>> &featLayerShapes) {    // 看自己用不用
        std::vector <int> clses = {};   // 以下三个向量到时做成插件后输出给画框时可以对得到的数据转类型
        std::vector <float> m_scores = {};
        std::vector <std::vector <std::vector <float>>> v_projs_regress = {};
        std::vector <int> xdata = {};      // 列
        std::vector <int> ydata = {};      // 行
        std::vector <float> xdataf = {};   // xdata与其他数据相加和结果用float
        std::vector <float> ydataf = {};   // ydata与其他数据相加和结果用float
        std::vector <std::vector <float>> lay2data = {};    // netout02的部分值
        std::vector <std::vector <float>> lay3data = {};    // netout03的部分值
        // 每个tensor中的四维数据在featLayerData中压缩成一维的
        std::shared_ptr<void> netout01;  // netout01、02、03、04都用于存放feetLayerData
        std::shared_ptr<void> netout02;
        std::shared_ptr<void> netout03;
        std::shared_ptr<void> netout04;
        float begindata[NETOUT01_DIM2][NETOUT01_DIM3][NETOUT01_DIM4];      // netout01的tensor数据
        float padbegindata[NETOUT01_DIM2][PAD + NETOUT01_DIM3 + PAD][PAD + NETOUT01_DIM4 + PAD];   // 池化时pad后的数据
        float pooldata[NETOUT01_DIM2][NETOUT01_DIM3][NETOUT01_DIM4];       // 池化后的结果
        float topdata[NETOUT01_DIM1 * NETOUT01_DIM2 * NETOUT01_DIM3 * NETOUT01_DIM4];              // 转化成一维数据
        int netindex = 0;                  // 索引
        float scores[TOPK_CANDIDATES];                 // 前一百的置信度
        int indices[TOPK_CANDIDATES];                  // 前一百的置信度的索引

        downdata(netout01, netout02, netout03, netout04, featLayerData);
        // 因为netout01是包含3个2维数组，池化操作对象是2维数组
        MaxPool(padbegindata, begindata, pooldata, netindex++, netout01);
        MaxPool(padbegindata, begindata, pooldata, netindex++, netout01);
        MaxPool(padbegindata, begindata, pooldata, netindex++, netout01);
        ObtainMainProj2d(begindata, topdata, indices, scores, xdata, ydata, clses, m_scores);
        DataFromSameIndex(lay2data, lay3data, netout02, netout03, xdata.size(), xdata, ydata);
        FrameCoordinates(xdata.size(), v_projs_regress, lay2data, xdataf, ydataf, xdata, ydata, lay3data);

        for(int i = 0; i < xdata.size(); ++i){
            for(int j = 0; j < FRAME_POINT; ++j){
                MxBase::ObjectInfo det;    // 将结果赋给ObjectInfo,用于画框等操作
                det.x0 = v_projs_regress[i][j][0];     // 用四个ObjectInfo变量来传输一个框的信息
                det.y0 = v_projs_regress[i][j][1];
                det.x1 = v_projs_regress[i][j + 1][0];
                det.y1 = v_projs_regress[i][j + 1][1];
                det.classId = clses[i];
                det.confidence = m_scores[i];
                detBoxes.emplace_back(det);
                j = j + 1;
            }
        }

    }

    extern "C" {
    std::shared_ptr <MxBase::RTM3DPostProcess> GetObjectInstance() {
        LogInfo << "Begin to get RTM3DPostProcess instance.";
        auto instance = std::make_shared<MxBase::RTM3DPostProcess>();
        LogInfo << "End to get RTM3DPostProcess instance.";
        return instance;
    }
    }
}