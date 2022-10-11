/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "MxpiX3DPreProcess.h"
using namespace MxPlugins;
using namespace MxTools;
using namespace MxBase;
using namespace std;

namespace
{
    const uint32_t YUV_BYTES_NU = 3;
    const uint32_t YUV_BYTES_DE = 2;
    const uint32_t TENSOR_LENGTH = 13;
    const uint32_t IMAGE_SHORT_LENGTH = 182;
    const uint32_t MAX_WINDOW_STRIDE = 28;
    const uint32_t SAMPLE_NUM = 3;
    const uint32_t HALF_DIV = 2;
    const uint32_t INPUT_CHANNEL = 3;
    cv::Mat visionQueue[SAMPLE_NUM*TENSOR_LENGTH*MAX_WINDOW_STRIDE];
    uint32_t visionQueueHeadIdx = 0;
    uint32_t visionQueueLength;
    uint32_t process_count = 0;
    uint32_t stride_count = 0;
    uint32_t stackFrameStartIdx = 0;
} // namespace

APP_ERROR MxpiX3DPreProcess::Init(
    std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    std::cout << "MxpiX3DPreProcess::Init start." << std::endl;
    std::shared_ptr<std::string> dataSource =
        std::static_pointer_cast<std::string>(configParamMap["dataSource"]);
    dataSource_ = *dataSource;
    std::shared_ptr<std::uint32_t> skipFrameNum =
        std::static_pointer_cast<std::uint32_t>(configParamMap["skipFrameNum"]);
    skipFrameNum_ = *skipFrameNum;
    std::shared_ptr<std::uint32_t> windowStride =
        std::static_pointer_cast<std::uint32_t>(configParamMap["windowStride"]);
    windowStride_ = *windowStride;

    visionQueueLength = skipFrameNum_ * (TENSOR_LENGTH-1)+1;
    return APP_ERR_OK;
}

APP_ERROR MxpiX3DPreProcess::DeInit()
{
    std::cout << "MxpiX3DPreProcess::DeInit end." << std::endl;
    return APP_ERR_OK;
}

APP_ERROR MxpiX3DPreProcess::CheckDataSource(MxTools::MxpiMetadataManager &mxpiMetadataManager)
{
    if (mxpiMetadataManager.GetMetadata(dataSource_) == nullptr)
    {
        LogDebug << GetError(APP_ERR_METADATA_IS_NULL, pluginName_)
                 << "data metadata is null. please check"
                 << "Your property dataSource(" << dataSource_ << ").";
        return APP_ERR_METADATA_IS_NULL;
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiX3DPreProcess::PreProcessVision(MxTools::MxpiVision srcMxpiVision,uint32_t insert_idx){
    auto& visionInfo = srcMxpiVision.visioninfo();
    auto& visionData = srcMxpiVision.visiondata();
    MxBase::MemoryData memorySrc = {};
    memorySrc.deviceId = visionData.deviceid();
    memorySrc.type = (MxBase::MemoryData::MemoryType) visionData.memtype();
    memorySrc.size = visionData.datasize();
    memorySrc.ptrData = (void*)visionData.dataptr();
    MxBase::MemoryData memoryDst(visionData.datasize(),
    MxBase::MemoryData::MEMORY_HOST_NEW);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, memorySrc); 
    if (ret != APP_ERR_OK) {
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }
    cv::Mat imgYuv = cv::Mat(visionInfo.heightaligned()  * YUV_BYTES_NU / YUV_BYTES_DE,
                            visionInfo.widthaligned(), CV_8UC1, memoryDst.ptrData);
    cv::Mat imgRgb = cv::Mat(visionInfo.heightaligned(), visionInfo.widthaligned(), CV_8UC3);
    cv::cvtColor(imgYuv, imgRgb, cv::COLOR_YUV2RGB_NV21);

    uint32_t h = imgRgb.size[0];
    uint32_t w = imgRgb.size[1];
    uint32_t h_resize,w_resize;
    cv::Rect clip0,clip1,clip2;
    if(h>w){
        w_resize = IMAGE_SHORT_LENGTH;
        h_resize = (uint32_t)h*1.0*w_resize/w;
        clip0 = cv::Rect(0, 0, IMAGE_SHORT_LENGTH, IMAGE_SHORT_LENGTH);
        clip1 = cv::Rect(0, (h_resize-IMAGE_SHORT_LENGTH)/HALF_DIV, IMAGE_SHORT_LENGTH, IMAGE_SHORT_LENGTH);
        clip2 = cv::Rect(0, h_resize-IMAGE_SHORT_LENGTH, IMAGE_SHORT_LENGTH, IMAGE_SHORT_LENGTH);
    }else{
        h_resize = IMAGE_SHORT_LENGTH;
        w_resize = (uint32_t)h_resize*1.0*w/h;
        clip0 = cv::Rect(0, 0, IMAGE_SHORT_LENGTH, IMAGE_SHORT_LENGTH);
        clip1 = cv::Rect((w_resize-IMAGE_SHORT_LENGTH)/HALF_DIV, 0, IMAGE_SHORT_LENGTH, IMAGE_SHORT_LENGTH);
        clip2 = cv::Rect(w_resize-IMAGE_SHORT_LENGTH, 0, IMAGE_SHORT_LENGTH, IMAGE_SHORT_LENGTH);
    }
    cv::resize(imgRgb, imgRgb, cv::Size(w_resize, h_resize), cv::INTER_LINEAR);
    cv::Mat dstImg0 = imgRgb(clip0).clone();
    cv::Mat dstImg1 = imgRgb(clip1).clone();
    cv::Mat dstImg2 = imgRgb(clip2).clone();
    visionQueue[insert_idx++] = dstImg0;
    visionQueue[insert_idx++] = dstImg1;
    visionQueue[insert_idx] = dstImg2;
    return APP_ERR_OK;
}

std::shared_ptr<MxpiTensorPackageList> MxpiX3DPreProcess::StackFrame(uint32_t start_idx){
    uint32_t idx;
    std::shared_ptr<MxpiTensorPackageList> tensorPackageList(new MxpiTensorPackageList,
                                                            g_deleteFuncMxpiTensorPackageList);
    cv::Mat sampleMat = visionQueue[start_idx*SAMPLE_NUM];
    uint32_t sampleMatSize = sampleMat.cols*sampleMat.rows*sampleMat.elemSize();
    for(uint32_t j=0;j<SAMPLE_NUM;j++){
        MxBase::MemoryData concatData = {};
        concatData.deviceId = deviceId_;
        concatData.type = (MxBase::MemoryData::MemoryType) 0;
        concatData.size = TENSOR_LENGTH*sampleMatSize;
        APP_ERROR ret = MxBase::MemoryHelper::MxbsMalloc(concatData);
        if (ret != APP_ERR_OK) {
            LogError << "MxbsMalloc failed";
            LogError << "concatData.size:" << concatData.size << " concatData.type:" << concatData.type
                        << "concatData.deviceId:" << concatData.deviceId;
            return tensorPackageList;
        }
        for(uint32_t i=0;i<TENSOR_LENGTH;i++){
            idx = (start_idx+i*skipFrameNum_)%visionQueueLength;
            MxBase::MemoryData newData = {};
            newData.deviceId = deviceId_;
            newData.type = (MxBase::MemoryData::MemoryType) 0;
            newData.size = (uint32_t) sampleMatSize;
            newData.ptrData = (void *) ((uint8_t*) concatData.ptrData + i * newData.size);
            MxBase::MemoryData memoryData = MemoryData(visionQueue[3*idx+j].data,sampleMatSize,MemoryData::MEMORY_HOST_MALLOC);
            ret = MxBase::MemoryHelper::MxbsMemcpy(newData, memoryData, memoryData.size);
            if (ret != APP_ERR_OK) {
                LogError << "MxbsMemcpy failed";
                MxBase::MemoryHelper::Free(concatData);
                return tensorPackageList;
            }
        }
        auto tensorPackage = tensorPackageList->add_tensorpackagevec();
        auto tensorVector = tensorPackage->add_tensorvec();
        tensorVector->set_tensordataptr((uint64) concatData.ptrData);
        tensorVector->set_tensordatasize(concatData.size);
        tensorVector->set_deviceid(concatData.deviceId);
        tensorVector->set_memtype((MxpiMemoryType) concatData.type);
        // explicitly specify tenspr shape
        tensorVector->add_tensorshape(TENSOR_LENGTH);
        tensorVector->add_tensorshape(IMAGE_SHORT_LENGTH);
        tensorVector->add_tensorshape(IMAGE_SHORT_LENGTH);
        tensorVector->add_tensorshape(INPUT_CHANNEL);
    }
    return tensorPackageList;
}

APP_ERROR MxpiX3DPreProcess::Process(std::vector<MxpiBuffer *> &mxpiBuffer)
{
    LogInfo << "Begin to process MxpiX3DPreProcess(" << elementName_ << ").";
    MxpiBuffer *inputMxpiBuffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*inputMxpiBuffer);
    APP_ERROR ret = CheckDataSource(mxpiMetadataManager);

    if (ret != APP_ERR_OK)
    {
        SendData(0, *inputMxpiBuffer);
        return ret;
    }
    std::shared_ptr<void> data_source = mxpiMetadataManager.GetMetadata(dataSource_);
    std::shared_ptr<MxpiVisionList> visionList = std::static_pointer_cast<MxpiVisionList>(data_source);
    
    PreProcessVision(visionList->visionvec(0),visionQueueHeadIdx*SAMPLE_NUM);
    visionQueueHeadIdx += 1;
    visionQueueHeadIdx = visionQueueHeadIdx % visionQueueLength;

    process_count +=1;
    if (process_count >= visionQueueLength)
    {
        if(stride_count%windowStride_==0){
            stride_count=0;
            auto tensorPackageListPtr = StackFrame(stackFrameStartIdx);
            stackFrameStartIdx+=windowStride_;
            stackFrameStartIdx%=visionQueueLength;

            const string metaKey = pluginName_;
            auto *outbuffer = MxpiBufferManager::CreateHostBuffer(inputParam);
            MxpiMetadataManager mxpiMetadataManager(*outbuffer);
            auto ret = mxpiMetadataManager.AddProtoMetadata(metaKey,
                                                            static_pointer_cast<void>(tensorPackageListPtr));
            if (ret != APP_ERR_OK) {
                LogError << ErrorInfo_.str();
                SendMxpiErrorInfo(*outbuffer, pluginName_, ret, ErrorInfo_.str());
                SendData(0, *outbuffer);
                return ret;
            }
            SendData(0, *outbuffer);
        }
        stride_count+=1;
    }

    MxpiBufferManager::DestroyBuffer(inputMxpiBuffer);
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiX3DPreProcess::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    auto dataSource = std::make_shared<ElementProperty<string>>(
        ElementProperty<string>{
        STRING, 
        "dataSource", 
        "dataSource", 
        "data source",
        "defalut", "NULL", "NULL"});
    auto skipFrameNum = std::make_shared<ElementProperty<uint>>(
        ElementProperty<uint>{
        UINT, 
        "skipFrameNum", 
        "skipFrameNum",
        "the number of skip frame",
        5, 1, 10});
    auto windowStride = std::make_shared<ElementProperty<uint>>(
        ElementProperty<uint>{
        UINT, 
        "windowStride", 
        "windowStride",
        "the number of window stride",
        1, 1, MAX_WINDOW_STRIDE});
    properties.push_back(dataSource);
    properties.push_back(skipFrameNum);
    properties.push_back(windowStride);
    return properties;
}

// Register the VpcResize plugin through macro
MX_PLUGIN_GENERATE(MxpiX3DPreProcess)
