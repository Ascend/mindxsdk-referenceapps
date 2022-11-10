/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "MxpiObjectFilter.h"
using namespace MxPlugins;
using namespace MxTools;
using namespace MxBase;
using namespace std;

namespace
{
    const uint32_t MIN_DIP = 32;
} // namespace

APP_ERROR MxpiObjectFilter::Init(
    std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    std::cout << "MxpiObjectFilter::Init start." << std::endl;
    std::shared_ptr<std::string> dataSource =
        std::static_pointer_cast<std::string>(configParamMap["dataSource"]);
    dataSource_ = *dataSource;
    return APP_ERR_OK;
}

APP_ERROR MxpiObjectFilter::DeInit()
{
    std::cout << "MxpiObjectFilter::DeInit end." << std::endl;
    return APP_ERR_OK;
}

APP_ERROR MxpiObjectFilter::CheckDataSource(MxTools::MxpiMetadataManager &mxpiMetadataManager)
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

APP_ERROR MxpiObjectFilter::Process(std::vector<MxpiBuffer *> &mxpiBuffer)
{
    LogInfo << "Begin to process MxpiObjectFilter(" << elementName_ << ").";
    MxpiBuffer *inputMxpiBuffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*inputMxpiBuffer);
    APP_ERROR ret = CheckDataSource(mxpiMetadataManager);
    if (ret != APP_ERR_OK)
    {
        SendData(0, *inputMxpiBuffer);
        return ret;
    }
    shared_ptr<void> srcObjectList = mxpiMetadataManager.GetMetadata(dataSource_);
    std::shared_ptr<MxpiObjectList> srcObjectListSptr = std::static_pointer_cast<MxpiObjectList>(srcObjectList);
    shared_ptr<MxpiObjectList> dstMxpiObjectListSptr = make_shared<MxpiObjectList>();
    for (uint32_t i = 0; i < srcObjectListSptr->objectvec_size(); i++)
    {
        auto srcMxpiObject = srcObjectListSptr->objectvec(i);
        if (srcMxpiObject.x1() - srcMxpiObject.x0() > MIN_DIP && srcMxpiObject.y1() - srcMxpiObject.y0() > MIN_DIP)
        {
            auto dstMxpiObject = dstMxpiObjectListSptr->add_objectvec();
            dstMxpiObject->set_x0(srcMxpiObject.x0());
            dstMxpiObject->set_y0(srcMxpiObject.y0());
            dstMxpiObject->set_x1(srcMxpiObject.x1());
            dstMxpiObject->set_y1(srcMxpiObject.y1());
            auto dstMxpiClass = dstMxpiObject->add_classvec();
            dstMxpiClass->set_confidence(srcMxpiObject.classvec(0).confidence());
            dstMxpiClass->set_classid(srcMxpiObject.classvec(0).classid());
            dstMxpiClass->set_classname(srcMxpiObject.classvec(0).classname());
        }
    }
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiObjectListSptr)); // Add Generated data to metedata
    if (ret != APP_ERR_OK)
    {
        SendData(0, *inputMxpiBuffer);
        return ret;
    }
    SendData(0, *inputMxpiBuffer);
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiObjectFilter::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    auto dataSource = std::make_shared<ElementProperty<string>>(
        ElementProperty<string> {
            STRING, "dataSource", "dataSource", "data source", "defalut", "NULL", "NULL"});
    properties.push_back(dataSource);
    return properties;
}

// Register the VpcResize plugin through macro
MX_PLUGIN_GENERATE(MxpiObjectFilter)
