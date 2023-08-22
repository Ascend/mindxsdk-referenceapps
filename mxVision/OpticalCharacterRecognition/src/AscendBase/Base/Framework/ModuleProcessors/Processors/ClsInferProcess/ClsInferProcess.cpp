#include "ClsInferProcess.h"
#include "ClsPostProcess/ClsPostProcess.h"
#include "Utils.h"

using namespace ascendOCR;

ClsInferProcess::ClsInferProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

ClsInferProcess::~ClsInferProcess() {}

APP_ERROR ClsInferProcess::Init(ConfigParser &configParser, ModuleInitParams &initParams)
{
    LogInfo << "Begin to init instance " << initParams.instanceId;
    InitParams(initParams);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "ClsInferProcess[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }
    LogInfo << "ClsInferProcess [" << instanceId_ << "]: Init success.";
    return ret;
}

APP_ERROR ClsInferProcess::DeInit(void)
{
    LogInfo << "ClsInferProcess [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR ClsInferProcess::ParseConfig(ConfigParser &configParser)
{
    std::vector<uint32_t> deviceIdVec;
    APP_ERROR ret = configParser.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret != APP_ERR_OK || deviceIdVec.empty()) {
        LogError << "Get device id failed, please check the value of deviceId";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    deviceId_ = (int32_t)deviceIdVec[instanceId_ % deviceIdVec.size()];
    LogDebug << "deviceId: " << deviceId_;

    std::string clsModelPath;
    ret = configParser.GetStringValue("clsModelPath", clsModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Get clsModelPath failed, please check the value of clsModelPath.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    ret = Utils::CheckPath(clsModelPath, "clsModelPath");
    if (ret != APP_ERR_OK) {
        LogError << "clsModelPath: " << clsModelPath << "is not exist or can not read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    LogDebug << "clsModelPath: " << clsModelPath;

    ClsNet_.reset(new MxBase::Model(clsModelPath, deviceId_));
    clsHeight = 48;
    clsWidth = 192;
    return APP_ERR_OK;
}

std::vector<MxBase::Tensor> ClsInferProcess::ClsModelInfer(uint8_t *srcData, uint32_t batchSize, int maxResizedW)
{
    LogDebug << "Infer: maxResizedW: " << maxResizedW << std::endl;

    std::vector<uint32_t> shape;
    shape.push_back(batchSize);
    shape.push_back(3);
    shape.push_back(clsHeight);
    shape.push_back(clsWidth);
    MxBase::TensorDType tensorDataType = MxBase::TensorDType::FLOAT32;

    std::vector<MxBase::Tensor> inputs = {};
    MxBase::Tensor imageToTensor(srcData, shape, tensorDataType, deviceId_);
    inputs.push_back(imageToTensor);

    LogDebug << "batchSize: " << batchSize;

    // start to inference
    auto inferStartTime = std::chrono::high_resolution_clock::now();
    ClsOutputs = ClsNet_->Infer(inputs);
    auto inferEndTime = std::chrono::high_resolution_clock::now();
    double inferCostTime = std::chrono::duration<double, std::milli>(inferEndTime - inferStartTime).count();
    Signal::clsInferTime += inferCostTime;
    for (auto &output : ClsOutputs) {
        output.ToHost();
    }
    LogInfo << "End Cls Model Infer Progress.";
    return ClsOutputs;
}

APP_ERROR ClsInferProcess::Process(std::shared_ptr<void> commonData)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    if (data->eof) {
        SendToNextModule(MT_ClsPostProcess, data, data->channelId);
        auto endTime = std::chrono::high_resolution_clock::now();
        double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        Signal::clsInferProcessTime += costTime;
        Signal::e2eProcessTime += costTime;
        return APP_ERR_OK;
    }

    std::vector<MxBase::Tensor> ClsOutput = ClsModelInfer(data->imgBuffer, data->batchSize, data->maxResizedW);
    data->outputTensorVec = ClsOutput;
    if (data->imgBuffer != nullptr) {
        delete data->imgBuffer;
        data->imgBuffer = nullptr;
    }
    SendToNextModule(MT_ClsPostProcess, data, data->channelId);

    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Signal::clsInferProcessTime += costTime;
    Signal::e2eProcessTime += costTime;
    return APP_ERR_OK;
}
