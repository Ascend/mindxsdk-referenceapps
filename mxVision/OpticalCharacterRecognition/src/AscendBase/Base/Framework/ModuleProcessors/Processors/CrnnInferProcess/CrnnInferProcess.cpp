#include "CrnnInferProcess.h"
#include "CrnnPostProcess/CrnnPostProcess.h"
#include "Utils.h"


using namespace ascendOCR;

CrnnInferProcess::CrnnInferProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

CrnnInferProcess::~CrnnInferProcess() {}

APP_ERROR CrnnInferProcess::Init(ConfigParser &configParser, ModuleInitParams &initParams)
{
    LogInfo << "Begin to init instance " << initParams.instanceId;
    InitParams(initParams);
    APP_ERROR ret = ParseConfig(configParser);
    if (ret != APP_ERR_OK) {
        LogError << "CrnnInferProcess[" << instanceId_ << "]: Fail to parse config params." << GetAppErrCodeInfo(ret);
        return ret;
    }
    LogInfo << "CrnnInferProcess [" << instanceId_ << "]: Init success.";
    return ret;
}

APP_ERROR CrnnInferProcess::DeInit(void)
{
    LogInfo << "CrnnInferProcess [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

APP_ERROR CrnnInferProcess::ParseConfig(ConfigParser &configParser)
{
    std::vector<uint32_t> deviceIdVec;
    APP_ERROR ret = configParser.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret != APP_ERR_OK || deviceIdVec.empty()) {
        LogError << "Get device id failed, please check the value of deviceId";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    deviceId_ = (int32_t)deviceIdVec[instanceId_ % deviceIdVec.size()];

    std::string recModelPath;
    ret = configParser.GetStringValue("recModelPath", recModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Get recModelPath failed, please check the value of recModelPath.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    ret = Utils::CheckPath(recModelPath, "recModelPath");
    if (ret != APP_ERR_OK) {
        LogError << "recModelPath: " << recModelPath << "is not exist or can not read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    LogDebug << "recModelPath: " << recModelPath;

    std::vector<std::string> files;
    Utils::GetAllFiles(recModelPath, files);
    for (auto &file : files) {
        crnnNet_.push_back(new MxBase::Model(file, deviceId_));
        std::vector<std::vector<uint64_t>> dynamicGearInfo = crnnNet_[crnnNet_.size() - 1]->GetDynamicGearInfo();
        mStdHeight = dynamicGearInfo[0][2];
        batchSizeList.push_back(dynamicGearInfo[0][0]);
    }
    std::sort(batchSizeList.begin(), batchSizeList.end(), Utils::UintCompare);
    std::sort(crnnNet_.begin(), crnnNet_.end(), Utils::ModelCompare);

    return APP_ERR_OK;
}

std::vector<MxBase::Tensor> CrnnInferProcess::CrnnModelInfer(uint8_t *srcData, uint32_t batchSize, int maxResizedW)
{
    LogDebug << "Infer: maxResizedW: " << maxResizedW << std::endl;

    std::vector<uint32_t> shape;
    shape.push_back(batchSize);
    shape.push_back(3);
    shape.push_back(mStdHeight);
    shape.push_back(maxResizedW);
    MxBase::TensorDType tensorDataType = MxBase::TensorDType::FLOAT32;

    std::vector<MxBase::Tensor> inputs = {};
    MxBase::Tensor imageToTensor(srcData, shape, tensorDataType, deviceId_);
    inputs.push_back(imageToTensor);

    // choose the model
    int modelIndex = 0;
    auto it = find(batchSizeList.begin(), batchSizeList.end(), batchSize);
    modelIndex = it - batchSizeList.begin();

    LogDebug << "batchSize: " << betchSize;
    LogDebug << "modelIndex: " << modelIndex;

    // start to inference
    auto inferStartTime = std::chrono::high_resolution_clock::now();
    crnnOutputs = crnnNet_[modelIndex]->Infer(inputs);
    auto inferEndTime = std::chrono::high_resolution_clock::now();
    double inferCostTime = std::chrono::duration<double, std::milli>(inferEndTime - inferStartTime).count();
    Signal::recInferTime += inferCostTime;
    for (auto &output : crnnOutputs) {
        output.ToHost();
    }
    LogInfo << "End Crnn Model Infer Progress.";
    return crnnOutputs;
}

APP_ERROR CrnnInferProcess::Process(std::shared_ptr<void> commonData)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    if (data->eof) {
        auto endTime = std::chrono::high_resolution_clock::now();
        double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        Signal::recInferProcessTime += CostTime;
        Signal::e2eProcessTime += costTime;
        SendToNextModule(MT_CrnnPostProcess, data, data->channelId);
        return APP_ERR_OK;
    }

    std::vector<MxBase::Tensor> crnnOutput = crnnModelInfer(data->imgBuffer, data->batchSize, data->maxResizedW);
    data->outputTensorVec = crnnOutput;
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    if (data->imgBuffer != nullptr) {
        delete data->imgBuffer;
        data->imgBuffer = nullptr;
    }

    Signal::recInferProcessTime += costTime;
    Signal::e2eProcessTime += costTime;
    SendToNextModule(MT_CrnnPostProcess, data, data->channelId);

    return APP_ERR_OK;
}
