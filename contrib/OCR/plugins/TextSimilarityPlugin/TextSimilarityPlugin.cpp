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
 
#include "TextSimilarityPlugin.h"
#include <iostream>
#include "MxBase/Log/Log.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include <mutex>
#include <thread>
#include <map>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <codecvt>
#include <algorithm>
#include <cstdint>
#include <istream>
#include <sstream>
using namespace MxBase;
using namespace MxTools;
using namespace MxPlugins;
using namespace std;

APP_ERROR TextSimilarityPlugin::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "Begin to initialize TextInfoPlugin(" << pluginName_ << ").";

    dataSource_ = *std::static_pointer_cast<std::string>(configParamMap["dataSource"]);

    LogInfo << "End to initialize MxpiFairmot(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR TextSimilarityPlugin::DeInit()
{
    LogInfo << "Begin to deinitialize MxpiFairmot(" << pluginName_ << ").";
    LogInfo << "End to deinitialize MxpiFairmot(" << pluginName_ << ").";
    return APP_ERR_OK;
}

void GetTensors(const std::shared_ptr<MxTools::MxpiTensorPackageList> &tensorPackageList,
                std::vector<MxBase::TensorBase> &tensors) {
    for (int i = 0; i < tensorPackageList->tensorpackagevec_size(); ++i) {
        for (int j = 0; j < tensorPackageList->tensorpackagevec(i).tensorvec_size(); j++) {
            MxBase::MemoryData memoryData = {};
            memoryData.deviceId = tensorPackageList->tensorpackagevec(i).tensorvec(j).deviceid();
            memoryData.type = (MxBase::MemoryData::MemoryType)tensorPackageList->
                    tensorpackagevec(i).tensorvec(j).memtype();
            memoryData.size = (uint32_t) tensorPackageList->
                    tensorpackagevec(i).tensorvec(j).tensordatasize();
            memoryData.ptrData = (void *) tensorPackageList->
                    tensorpackagevec(i).tensorvec(j).tensordataptr();
            std::vector<uint32_t> outputShape = {};
            for (int k = 0; k < tensorPackageList->
                    tensorpackagevec(i).tensorvec(j).tensorshape_size(); ++k) {
                outputShape.push_back((uint32_t) tensorPackageList->
                        tensorpackagevec(i).tensorvec(j).tensorshape(k));
            }
            MxBase::TensorBase tmpTensor(memoryData, true, outputShape,
                                         (MxBase::TensorDataType)tensorPackageList->
                                                 tensorpackagevec(i).tensorvec(j).tensordatatype());
            tensors.push_back(tmpTensor);
        }
    }
}

std::vector<std::shared_ptr<void>> TextSimilarityPlugin::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    auto datasource = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
            STRING,
            "dataSource",
            "dataSource",
            "the name of cropped image source",
            "default", "NULL", "NULL"
    });
    properties.push_back(datasource);
    return properties;
}

MxpiPortInfo TextSimilarityPlugin::DefineInputPorts()
{
    MxpiPortInfo inputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}, {"ANY"}, {"ANY"}, {"ANY"}, {"ANY"}, {"ANY"}};
    GenerateStaticInputPortsInfo(value, inputPortInfo);
    return inputPortInfo;
}

MxpiPortInfo TextSimilarityPlugin::DefineOutputPorts()
{
    MxpiPortInfo outputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticOutputPortsInfo(value, outputPortInfo);
    return outputPortInfo;
}

namespace {
    MX_PLUGIN_GENERATE(TextSimilarityPlugin)
}

void Covert(const std::shared_ptr<MxTools::MxpiTextsInfoList> &textsInfoList,
            std::vector<MxBase::TextsInfo> &textsInfoVec)
{
    for (uint32_t i = 0; i < textsInfoList->textsinfovec_size(); i++) {
        auto textsInfo = textsInfoList->textsinfovec(i);
        MxBase::TextsInfo text;
        for (uint32_t j = 0; j < textsInfo.text_size(); j++) {
            auto textInfo = textsInfo.text(j);
            if (textInfo == ""){
                continue;
            }
            text.text.push_back(textInfo);
        }
        textsInfoVec.push_back(text);
    }
}

APP_ERROR TextSimilarityPlugin::Process(std::vector<MxpiBuffer *> &mxpiBuffer)
{
    /*
     * get the MxpiVisionList and MxpiTrackletList
     * */
    LogInfo << "Begin to process MxpiMotSimpleSort(" << elementName_ << ").";
    // Get MxpiVisionList and MxpiTrackletList from mxpibuffer
    MxpiBuffer *inputMxpiBuffer0 = mxpiBuffer[0];   // deviceID[0]
    MxpiMetadataManager mxpiMetadataManager(*inputMxpiBuffer0);

    vector<string> names;
    std::stringstream ss(dataSource_); // Turn the std::string into a stream.
    std::string tok;

    while (getline(ss, tok, ','))
    {
        names.push_back(tok);
    }

    // Get the metadata from buffer
    std::shared_ptr<void> metadata0 = mxpiMetadataManager.GetMetadata(names[0]);
    std::shared_ptr<MxpiTensorPackageList> srcTensorPackageListSptr0 =
            std::static_pointer_cast<MxpiTensorPackageList>(metadata0);

    // Get tensorbase
    std::vector<MxBase::TensorBase> tensors0 = {};
    GetTensors(srcTensorPackageListSptr0, tensors0);

    auto shape0 = tensors0[0].GetShape();
    std::vector<std::vector<float> > input1(shape0[1],std::vector<float>(shape0[2]));
    void *idPtr0 =  tensors0[0].GetBuffer();
    for(uint32_t i = 0; i < shape0[0]; i++) {
        for (uint32_t j = 0; j < shape0[1]; j++) {
            for(int k = 0;k < shape0[2];k++){
                float x0 = *((float *) idPtr0 + k+j*shape0[2]);
                input1[j][k] = x0;
            }
        }
    }

    MxpiBuffer *inputMxpiBuffer1 = mxpiBuffer[1];   // deviceID[0]
    MxpiMetadataManager mxpiMetadataManager1(*inputMxpiBuffer1);

    // Get the metadata from buffer
    std::shared_ptr<void> metadata1 = mxpiMetadataManager1.GetMetadata(names[1]);
    std::shared_ptr<MxpiTensorPackageList> srcTensorPackageListSptr1 =
            std::static_pointer_cast<MxpiTensorPackageList>(metadata1);

    // Get tensorbase
    std::vector<MxBase::TensorBase> tensors1 = {};
    GetTensors(srcTensorPackageListSptr1, tensors1);
    auto shape1 = tensors1[0].GetShape();
    std::vector<std::vector<float> > input2(shape1[1],std::vector<float>(shape1[2]));
    void *idPtr1 =  tensors1[0].GetBuffer();
    for(uint32_t i = 0; i < shape1[0]; i++) {
        for (uint32_t j = 0; j < shape1[1]; j++) {
            for(int k = 0;k < shape1[2];k++){
                float x0 = *((float *) idPtr1 + k+j*shape1[2]);
                input2[j][k] = x0;
            }
        }
    }

    // Get MxpiVisionList and MxpiTrackletList from mxpibuffer
    MxpiBuffer *inputMxpiBuffer2 = mxpiBuffer[2];   // deviceID[0]
    MxpiMetadataManager mxpiMetadataManager2(*inputMxpiBuffer2);

    // Get the metadata from buffer
    std::shared_ptr<void> metadata2 = mxpiMetadataManager2.GetMetadata(names[2]);
    std::shared_ptr<MxpiTensorPackageList> srcTensorPackageListSptr2 =
            std::static_pointer_cast<MxpiTensorPackageList>(metadata2);

    // Get tensorbase
    std::vector<MxBase::TensorBase> tensors2 = {};
    GetTensors(srcTensorPackageListSptr2, tensors2);
    auto shape2 = tensors2[0].GetShape();
    void *idPtr2 =  tensors2[0].GetBuffer();
    int  length1 = *(int *) idPtr2;

    // Get MxpiVisionList and MxpiTrackletList from mxpibuffer
    MxpiBuffer *inputMxpiBuffer3 = mxpiBuffer[3];   // deviceID[0]
    MxpiMetadataManager mxpiMetadataManager3(*inputMxpiBuffer3);

    // Get the metadata from buffer
    std::shared_ptr<void> metadata3 = mxpiMetadataManager3.GetMetadata(names[3]);
    std::shared_ptr<MxpiTensorPackageList> srcTensorPackageListSptr3 =
            std::static_pointer_cast<MxpiTensorPackageList>(metadata3);

    // Get tensorbase
    std::vector<MxBase::TensorBase> tensors3 = {};
    GetTensors(srcTensorPackageListSptr3, tensors3);
    auto shape3 = tensors3[0].GetShape();
    void *idPtr3 =  tensors3[0].GetBuffer();
    int  length2 = *(int *) idPtr3;

    // Get MxpiVisionList and MxpiTrackletList from mxpibuffer
    MxpiBuffer *inputMxpiBuffer4 = mxpiBuffer[4];   // deviceID[0]
    MxpiMetadataManager mxpiMetadataManager4(*inputMxpiBuffer4);

    // Get the metadata from buffer
    std::shared_ptr<void> metadata4 = mxpiMetadataManager4.GetMetadata(names[4]);
    std::shared_ptr<MxTools::MxpiTextsInfoList> mxpiTextsInfoList4 =
            std::static_pointer_cast<MxpiTextsInfoList>(metadata4);
    std::vector<MxBase::TextsInfo> textsInfoVec0 = {};
    Covert(mxpiTextsInfoList4, textsInfoVec0);

    // Get the metadata from buffer
    MxpiBuffer *inputMxpiBuffer5 = mxpiBuffer[5];   // deviceID[0]
    MxpiMetadataManager mxpiMetadataManager5(*inputMxpiBuffer5);
    std::shared_ptr<void> metadata5 = mxpiMetadataManager5.GetMetadata(names[5]);
    std::shared_ptr<MxTools::MxpiTextsInfoList> mxpiTextsInfoList5 =
            std::static_pointer_cast<MxpiTextsInfoList>(metadata5);
    std::vector<MxBase::TextsInfo> textsInfoVec1 = {};
    Covert(mxpiTextsInfoList5, textsInfoVec1);
    bool has_kay = false;
    float thresh = 0.7;
    for(int i = 1; i< length1 - 1; i++) {
        for(int j = 1; j < length2 - 1; j++) {
            float temp = similarity(input1[i],input2[j]);
            LogInfo << "text:(" << textsInfoVec0[0].text[i - 1]
                    << ") keyword:(" << textsInfoVec1[0].text[j - 1] << ") similarity:" << temp;
            if (temp > thresh) {
                has_kay = true;
            }
        }
    }

    LogInfo << "has key(bool)?:" << has_kay;
    // Send the data to downstream plugin
    SendData(0, *inputMxpiBuffer0);
    LogInfo << "End to process TextInfoPlugin(" << elementName_ << ").";
    return APP_ERR_OK;
}

float TextSimilarityPlugin::scalar_product(vector<float> a, vector<float> b)
{
    float product = 0;
    for (int i = 0; i <= a.size() - 1; i++){
        product = product + (a[i]) * (b[i]);
    }
    return product;
}

float TextSimilarityPlugin::linalg(vector<float> a) {
    float res = 0;
    for (int i = 0; i < a.size(); i++) {
        res = res + a[i] * a[i];
    }
    res = sqrt(res);
    return res;
}

float TextSimilarityPlugin::similarity(vector<float>& a, vector<float>& b) {
    return scalar_product(a, b) / (linalg(a) * linalg(b));
}