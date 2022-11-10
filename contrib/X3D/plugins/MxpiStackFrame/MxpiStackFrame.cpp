/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include "MxpiStackFrame.h"

using namespace MxBase;
using namespace MxTools;
using namespace MxPlugins;
using namespace std;

namespace
{
    const uint32_t TIME_FRAME_LENGTH = 13;
    const uint32_t X3D_INPUT_H = 192;
    const uint32_t X3D_INPUT_W = 192;
    const uint32_t X3D_INPUT_C = 3;
    const uint32_t MILLI_BASE = 1000;
    const float REUSE_RATIO = 2;
    bool isStop_ = true;
    std::map<uint32_t, uint32_t> trackCount;
    std::shared_ptr<BlockingMap> ObjectFrame = std::make_shared<BlockingMap>();
}

APP_ERROR MxpiStackFrame::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "Begin to initialize MxpiStackFrame(" << pluginName_ << ").";
    std::shared_ptr<std::string> visionSource = std::static_pointer_cast<std::string>(configParamMap["visionSource"]);
    visionsource_ = *visionSource;

    std::shared_ptr<std::string> trackSource = std::static_pointer_cast<std::string>(configParamMap["trackSource"]);
    tracksource_ = *trackSource;

    std::shared_ptr<std::uint32_t> framenum = std::static_pointer_cast<std::uint32_t>(configParamMap["frameNum"]);
    skipFrameNum_ = *framenum;

    std::shared_ptr<std::double_t> timeout = std::static_pointer_cast<double_t>(configParamMap["timeOut"]);
    timeout_ = *timeout;

    std::shared_ptr<std::uint32_t> sleeptime = std::static_pointer_cast<uint32_t>(configParamMap["sleepTime"]);
    sleepTime_ = *sleeptime;

    isStop_ = false;
    // crate CheckFrame thread in Init function
    CreateThread();
    LogInfo << "End to initialize MxpiStackFrame(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MxpiStackFrame::DeInit()
{
    LogInfo << "Begin to deinitialize MxpiStackFrame(" << pluginName_ << ").";
    // Block the current thread until the thread identified by *this ends its execution
    stopFlag_ = true;
    if (thread_->joinable())
    {
        thread_->join();
    }
    ObjectFrame.reset();
    LogInfo << "End to deinitialize MxpiStackFrame(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR MxpiStackFrame::CheckDataSource(MxTools::MxpiMetadataManager &mxpiMetadataManager)
{
    if (mxpiMetadataManager.GetMetadata(visionsource_) == nullptr)
    {
        LogDebug << GetError(APP_ERR_METADATA_IS_NULL, pluginName_)
                 << "vision metadata is null. please check"
                 << "Your property visionSource(" << visionsource_ << ").";
        return APP_ERR_METADATA_IS_NULL;
    }
    if (mxpiMetadataManager.GetMetadata(tracksource_) == nullptr)
    {
        LogDebug << GetError(APP_ERR_METADATA_IS_NULL, pluginName_)
                 << "track metadata is null. please check"
                 << "Your property dataSourceFeature(" << tracksource_ << ").";
        return APP_ERR_METADATA_IS_NULL;
    }
    return APP_ERR_OK;
}

void MxpiStackFrame::StackFrame(const std::shared_ptr<MxTools::MxpiVisionList> &visionList,
                                const std::shared_ptr<MxTools::MxpiTrackLetList> &trackLetList,
                                std::shared_ptr<BlockingMap> &blockingMap)
{
    for (int32_t i = 0; i < (int32_t)trackLetList->trackletvec_size(); i++)
    {
        auto &trackObject = trackLetList->trackletvec(i);
        uint32_t trackId = trackObject.trackid();
        // lost object has no header vector; no object ;
        if (trackObject.headervec_size() == 0 || visionList->visionvec_size() == 0)
        {
            continue;
        }
        // get the visionvec by memberid
        int32_t memberId = trackObject.headervec(0).memberid();
        // filter out images not cropped
        int32_t j = 0;
        for (; j < visionList->visionvec_size(); j++)
        {
            int32_t visionMemberId = visionList->visionvec(j).headervec(0).memberid();
            if (visionMemberId == memberId)
            {
                break;
            }
        }
        if (j >= visionList->visionvec_size())
        {
            continue;
        }
        auto vision = visionList->visionvec(memberId);
        // convert visiondata to memorydata
        MxBase::MemoryData memoryData = ConvertMemoryData(vision.visiondata());
        if (blockingMap->count(trackId) == 0)
        {
            // Initialize a new Mxpivisionlist
            blockingMap->Insert(trackId, memoryData);
            trackCount[trackId] = 1;
        }
        else
        {
            auto time_visionlist = blockingMap->Get(trackId);
            LogInfo << "time_visionlist.second->visionvec_size() " << time_visionlist.second->visionvec_size();
            if (time_visionlist.second->visionvec_size() >= TIME_FRAME_LENGTH)
            {
                continue;
            }
            else
            {
                uint32_t check_count = trackCount[trackId];
                if (check_count >= skipFrameNum_)
                {
                    blockingMap->Update(trackId, memoryData);
                    check_count = 1;
                }
                else
                {
                    check_count += 1;
                }
                trackCount[trackId] = check_count;
            }
        }
    }
    visionList->clear_visionvec();
}

void MxpiStackFrame::CreateThread()
{
    thread_.reset(new std::thread(&MxpiStackFrame::WatchThread, this));
}

void MxpiStackFrame::WatchThread()
{
    // set current device context ; same as deviceId in pipeline
    DeviceContext context = {};
    context.devId = 0;
    APP_ERROR ret = DeviceManager::GetInstance()->SetDevice(context);
    if (ret != APP_ERR_OK)
    {
        LogError << "SetDevice failed";
    }
    while (!stopFlag_)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime_));
        CheckFrames(ObjectFrame);
    }
}

std::shared_ptr<MxpiTensorPackageList> MxpiStackFrame::ConvertVisionList2TensorPackageList(
    std::shared_ptr<MxpiVisionList> &mxpiVisionList)
{
    // use make_shared with caution
    std::shared_ptr<MxpiTensorPackageList> tensorPackageList(new MxpiTensorPackageList,
                                                             g_deleteFuncMxpiTensorPackageList);
    MxBase::MemoryData concatData = {};
    concatData.deviceId = mxpiVisionList->visionvec(0).visiondata().deviceid();
    concatData.type = (MxBase::MemoryData::MemoryType)0;
    concatData.size =
        mxpiVisionList->visionvec_size() * (uint32_t)mxpiVisionList->visionvec(0).visiondata().datasize();
    // malloc new memory
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMalloc(concatData);
    if (ret != APP_ERR_OK)
    {
        LogError << "MxbsMalloc failed";
        LogError << "concatData.size:" << concatData.size << " concatData.type:" << concatData.type
                 << "concatData.deviceId:" << concatData.deviceId;
        return tensorPackageList;
    }
    for (int i = 0; i < mxpiVisionList->visionvec_size(); i++)
    {
        MxBase::MemoryData memoryData = ConvertMemoryData(mxpiVisionList->visionvec(i).visiondata());
        // memory copy
        MxBase::MemoryData newData = {};
        newData.deviceId = mxpiVisionList->visionvec(i).visiondata().deviceid();
        newData.type = (MxBase::MemoryData::MemoryType)0;
        newData.size = (uint32_t)mxpiVisionList->visionvec(i).visiondata().datasize();
        newData.ptrData = (void *)((uint8_t *)concatData.ptrData + i * newData.size);
        // MxbsMallocAndCopy cannot be used here, cause npu memory leak
        ret = MxBase::MemoryHelper::MxbsMemcpy(newData, memoryData, memoryData.size);
        if (ret != APP_ERR_OK)
        {
            LogError << "MxbsMemcpy failed";
            MxBase::MemoryHelper::Free(concatData);
            return tensorPackageList;
        }
    }
    auto tensorPackage = tensorPackageList->add_tensorpackagevec();
    auto tensorVector = tensorPackage->add_tensorvec();
    tensorVector->set_tensordataptr((uint64)concatData.ptrData);
    tensorVector->set_tensordatasize(concatData.size);
    tensorVector->set_deviceid(concatData.deviceId);
    tensorVector->set_memtype((MxpiMemoryType)concatData.type);
    // explicitly specify tenspr shape
    tensorVector->add_tensorshape(TIME_FRAME_LENGTH);
    tensorVector->add_tensorshape(X3D_INPUT_H);
    tensorVector->add_tensorshape(X3D_INPUT_W);
    tensorVector->add_tensorshape(X3D_INPUT_C);
    return tensorPackageList;
}

void MxpiStackFrame::CheckFrames(std::shared_ptr<BlockingMap> &blockingMap)
{
    // Get current timestamp
    LogInfo << "Begin to check frames";
    using Time = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<float>;
    using Millisecond = std::chrono::duration<double, std::ratio<1, MILLI_BASE>>;
    auto currentTime = Time::now();
    std::vector<uint32_t> keys = blockingMap->Keys();
    for (uint32_t key : keys)
    { // key <-> trackId
        auto TimeMxpiVisionList = blockingMap->Get(key);
        if (TimeMxpiVisionList.second == nullptr)
        {
            continue;
        }
        Duration duration = currentTime - TimeMxpiVisionList.first;
        double lastTime = std::chrono::duration_cast<Millisecond>(duration).count();
        if (lastTime > timeout_)
        {
            LogInfo << "Object:" << key << " is TimeOut";
            blockingMap->Clear(key);
            continue;
        }
        if (TimeMxpiVisionList.second->visionvec_size() == TIME_FRAME_LENGTH)
        {
            // Add MxpiTensorPackageList to metadata and Send data to downstream plugin
            const string metaKey = pluginName_;
            auto dstMxpiVisionLitsSptr = TimeMxpiVisionList.second;
            auto tensorPackageListPtr = ConvertVisionList2TensorPackageList(dstMxpiVisionLitsSptr);
            blockingMap->Clear(key);
            // sliding window
            std::vector<MxTools::MxpiVisionData> slidingWindow = {};
            uint32_t size = dstMxpiVisionLitsSptr->visionvec_size();
            for (uint32_t i = 0; i < size; i++)
            {
                auto mxpiVision = dstMxpiVisionLitsSptr->visionvec(i);
                auto mxpiVisionData = mxpiVision.visiondata();
                if (i < size / REUSE_RATIO)
                {
                    MxBase::MemoryData memoryData = ConvertMemoryData(mxpiVisionData);
                    APP_ERROR ret = MemoryHelper::MxbsFree(memoryData);
                    if (ret != APP_ERR_OK)
                    {
                        LogError << "MxbsFree failed";
                    }
                }
                else
                {
                    slidingWindow.emplace_back(mxpiVisionData);
                }
            }
            dstMxpiVisionLitsSptr->clear_visionvec();
            auto mxpiVisionList = ConstructMxpiVisionList(slidingWindow);
            blockingMap->Reinsert(key, mxpiVisionList);
            // CreateDeviceBuffer; need inputParam
            auto *outbuffer = MxpiBufferManager::CreateHostBuffer(inputParam);
            MxpiMetadataManager mxpiMetadataManager(*outbuffer);
            auto ret = mxpiMetadataManager.AddProtoMetadata(metaKey,
                                                            static_pointer_cast<void>(tensorPackageListPtr));
            if (ret != APP_ERR_OK)
            {
                LogError << ErrorInfo_.str();
                SendMxpiErrorInfo(*outbuffer, pluginName_, ret, ErrorInfo_.str());
                SendData(0, *outbuffer);
            }
            LogInfo << "Object:" << key << " has stacked enough frames and begin to send data to downstream plugin";
            SendData(0, *outbuffer);
        }
    }
}

std::shared_ptr<MxTools::MxpiVisionList> MxpiStackFrame::ConstructMxpiVisionList(
    std::vector<MxTools::MxpiVisionData> &slidingWindow)
{
    std::shared_ptr<MxTools::MxpiVisionList> dstMxpiVisionListSptr(new MxTools::MxpiVisionList,
                                                                   MxTools::g_deleteFuncMxpiVisionList);
    for (auto iter = slidingWindow.begin(); iter != slidingWindow.end(); iter++)
    {
        MxTools::MxpiVision *dstMxpivision = dstMxpiVisionListSptr->add_visionvec();
        MxTools::MxpiVisionInfo *mxpiVisionInfo = dstMxpivision->mutable_visioninfo();
        mxpiVisionInfo->set_format(1);
        mxpiVisionInfo->set_height(X3D_INPUT_H);
        mxpiVisionInfo->set_width(X3D_INPUT_W);
        mxpiVisionInfo->set_heightaligned(X3D_INPUT_H);
        mxpiVisionInfo->set_widthaligned(X3D_INPUT_W);
        // set MxpiVisionData by MemoryData
        MxTools::MxpiVisionData *mxpiVisionData = dstMxpivision->mutable_visiondata();
        mxpiVisionData->set_dataptr((uint64_t)iter->dataptr());
        mxpiVisionData->set_datasize(iter->datasize());
        mxpiVisionData->set_deviceid(iter->deviceid());
        mxpiVisionData->set_memtype(iter->memtype());
    }
    return dstMxpiVisionListSptr;
}

MxBase::MemoryData MxpiStackFrame::ConvertMemoryData(const MxTools::MxpiVisionData &mxpiVisionData)
{
    MxBase::MemoryData memoryData = {};
    memoryData.deviceId = mxpiVisionData.deviceid();
    memoryData.type = (MxBase::MemoryData::MemoryType)1;
    memoryData.size = (uint32_t)mxpiVisionData.datasize();
    memoryData.ptrData = (void *)mxpiVisionData.dataptr();
    return memoryData;
}

APP_ERROR MxpiStackFrame::Process(std::vector<MxpiBuffer *> &mxpiBuffer)
{
    LogInfo << "Begin to process MxpiStackFrame(" << elementName_ << ")";
    // Get MxpiVisionList and MxpiTrackletList from mxpibuffer
    MxpiBuffer *inputMxpiBuffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*inputMxpiBuffer);
    ErrorInfo_.str("");
    // check data source
    APP_ERROR ret = CheckDataSource(mxpiMetadataManager);
    if (ret != APP_ERR_OK)
    {
        SendData(0, *inputMxpiBuffer);
        return ret;
    }
    // Get metadata by key
    std::shared_ptr<void> vision_metadata = mxpiMetadataManager.GetMetadata(visionsource_);
    std::shared_ptr<MxpiVisionList> srcVisionListPtr = std::static_pointer_cast<MxpiVisionList>(vision_metadata);
    std::shared_ptr<void> track_metadata = mxpiMetadataManager.GetMetadata(tracksource_);
    std::shared_ptr<MxpiTrackLetList> srcTrackLetListPtr = std::static_pointer_cast<MxpiTrackLetList>(track_metadata);
    // Stacking frames by track ID ; Choose skipFrameNum_ to skip frame
    // DestroyBuffer (Get buffer to Host if needed)
    StackFrame(srcVisionListPtr, srcTrackLetListPtr, ObjectFrame);
    MxpiBufferManager::DestroyBuffer(inputMxpiBuffer);
    LogInfo << "End to process MxpiStackFrame(" << elementName_ << ").";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiStackFrame::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    // the cropped image
    auto visionsource = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "visionSource", "imageSource", "the name of cropped image source", "default", "NULL", "NULL"});
    // the tracklet information
    auto tracksource = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "trackSource", "objectSource", "tracklelist to get the responding id", "defalut", "NULL", "NULL"});
    // skip frame number
    auto framenum = std::make_shared<ElementProperty<uint>>(ElementProperty<uint> {
        UINT, "frameNum", "frameNum", "the number of skip frame", 1, 1, 10});
    auto timeout = std::make_shared<ElementProperty<std::double_t>>(ElementProperty<std::double_t> {
        DOUBLE, "timeOut", "timeOut", "Time to discard the frames", 5000., 500., 10000.});
    auto sleeptime = std::make_shared<ElementProperty<uint>>(ElementProperty<uint> {
        UINT, "sleepTime", "sleepTime", "The Time CheckFrames thread hangs", 100, 100, 1000});
    properties.push_back(visionsource);
    properties.push_back(tracksource);
    properties.push_back(framenum);
    properties.push_back(timeout);
    properties.push_back(sleeptime);
    return properties;
}

MxpiPortInfo MxpiStackFrame::DefineInputPorts()
{
    MxpiPortInfo inputPortInfo;
    // Input: {{Mxpivisionlist}, {MxpiTrackLetList}}
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticInputPortsInfo(value, inputPortInfo);
    return inputPortInfo;
}

MxpiPortInfo MxpiStackFrame::DefineOutputPorts()
{
    MxpiPortInfo outputPortInfo;
    // Output: {{MxpiTensorPackageList}}
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticOutputPortsInfo(value, outputPortInfo);
    return outputPortInfo;
}

namespace
{
    MX_PLUGIN_GENERATE(MxpiStackFrame)
}
