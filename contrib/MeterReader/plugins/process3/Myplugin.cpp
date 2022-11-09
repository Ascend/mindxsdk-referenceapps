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

#include "Myplugin.h"
#include "MxBase/Log/Log.h"
#include "postprocess.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace MxPlugins;
using namespace MxTools;
using namespace std;

const int imgSize = 512;
const int param_4 = 4;

APP_ERROR Myplugin::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "Myplugin::Init start.";
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    return APP_ERR_OK;
}

APP_ERROR Myplugin::DeInit()
{
    LogInfo << "Myplugin::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR Myplugin::Process(std::vector<MxpiBuffer*>& mxpiBuffer)
{
    LogInfo << "Myplugin::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "Myplugin process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "Myplugin process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the data from buffer
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }

    // 此处需要从metadata中读取出Deeplabv3后处理插件得到的输出，参考类型为MxpilmageMaskList
    shared_ptr<MxpiImageMaskList> srcImageMaskListptr = std::static_pointer_cast<MxpiImageMaskList>(metadata);

    int meter_num = srcImageMaskListptr->imagemaskvec_size();
    int length = srcImageMaskListptr->imagemaskvec(0).shape()[0] * srcImageMaskListptr->imagemaskvec(0).shape()[1];
    std::vector<std::vector<int64_t>> seg_result(meter_num, std::vector<int64_t>(length));
    std::vector<std::vector<int64_t>> seg_result_resize(meter_num, std::vector<int64_t>(imgSize * imgSize));

    for (int num = 0; num < meter_num; ++num) {
        const string single_seg_result = srcImageMaskListptr->imagemaskvec(num).datastr();
        cv::Mat kernel(param_4, param_4, CV_8U, cv::Scalar(1));
        std::vector<uint8_t> label_map(
            single_seg_result.begin(),
            single_seg_result.end());

        cv::Mat mask(srcImageMaskListptr->imagemaskvec(num).shape()[0],
            srcImageMaskListptr->imagemaskvec(num).shape()[1],
            CV_8UC1,
            label_map.data());
        cv::Size ResImgSiz = cv::Size(512, 512);
        cv::resize(mask, mask, ResImgSiz, 0, 0, cv::INTER_CUBIC);
        cv::erode(mask, mask, kernel);

        std::vector<int64_t> map;

        if (mask.isContinuous()) {
            map.assign(mask.data, mask.data + mask.total() * mask.channels());
        }
        else {
            for (int r = 0; r < mask.rows; r++) {
                map.insert(map.end(),
                    mask.ptr<int64_t>(r),
                    mask.ptr<int64_t>(r) + mask.cols * mask.channels());
            }
        }

        seg_result_resize[num] = std::move(map);
    }

    // 根据seg_result,调用postprocess函数，得到读取结果
    std::vector<READ_RESULT> read_results(meter_num); // 此处在postprocess.h文件内有定义READ_RESULT

    read_process(seg_result_resize, &read_results, meter_num);

    // 处理读取结果，暂时只进行结果的打印
    float ans;
    for (int i = 0; i < meter_num; i++) {
        // Provide a digital readout according to point location relative
        // to the scales
        float result = 0;
        if (read_results[i].scale_num > TYPE_THRESHOLD) {
            result = read_results[i].scales * (25.0f / 50.0f);
        }
        else {
            result = read_results[i].scales * (1.6f / 32.0f);
        }
        ans = result;
    }

    // 输出结果为float类型，识别结果
    shared_ptr<MxTools::MxpiClass> ans_ptr =
        make_shared<MxTools::MxpiClass>();
    ans_ptr->set_confidence(ans);

    LogInfo << ans_ptr->confidence();

    APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, std::static_pointer_cast<void>(ans_ptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "Myplugin add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }

    SendData(0, *buffer);
    LogInfo << "Myplugin::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> Myplugin::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;

    auto parentNameProSptr = (std::make_shared<ElementProperty<string>>)(ElementProperty<string> {
        STRING, "dataSource", "parentName", "the name of previous plugin", "mxpi_modelinfer0", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);

    return properties;
}

APP_ERROR Myplugin::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
    const MxpiErrorInfo mxpiErrorInfo)
{
    APP_ERROR ret = APP_ERR_OK;

    MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(pluginName, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(Myplugin)
