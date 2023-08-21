#include "ClsPostProcess.h"
#include "CrnnPreProcess/CrnnPreProcess.h"
#include "MxBase/MxBase.h"
#include "Utils.h"

using namespace ascendOCR;

ClsPostProcess::ClsPostProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

ClsPostProcess::~ClsPostProcess() {}

APP_ERROR ClsPostProcess::Init(ConfigParser &configParser, ModuleInitParams &initParams)
{
    LogInfo << "Begin to init instance " << initParams.instanceId;

    InitParams(initParams);
    LogInfo << "ClsPostProcess[" instanceId_ << "]: Init success.";
    return APP_ERR_OK;
}

APP_ERROR ClsPostProcess::DeInit(void)
{
    LogInfo << "ClsPostProcess[" instanceId_ << "]: DeInit success.";
    return APP_ERR_OK;
}

APP_ERROR ClsPostProcess::PostProcessCls(uint32_t frameSize, std::vector<MxBase::Tensor> &inferOutput,
    std::vector<std::string> &textsInfos)
{
    std::vector<uint32_t> shape = inferOutput[0].GetShape();
    auto *tensorData = (float *)inferOutput[0].GetData();
    uint32_t dirVecSize = shape[1];

    for (uint32_t i = 0; i < frameSize; i++) {
        uint32_t index = i * dirVecSize + 1;
        if (tensorData[index] > 0.9) {
            cv::rotate(imgMatVec[i], imgMatVec[i], cv::ROTATE_180);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR ClsPostProcess::Process(std::shared_ptr<void> commonData)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    
    if (!data->eof) {
        APP_ERROR ref = PostProcessCls(data->frameSize, data->outputTensorVec, data->imgMatVec);
        if (ret != APP_ERR_OK) {
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    SendToNextModule(MT_CollectProcess, data, data->channelId);

    Signal::clsPostProcessTime += costTime;
    Signal::e2eProcessTime += costTime;
    return APP_ERR_OK;
}