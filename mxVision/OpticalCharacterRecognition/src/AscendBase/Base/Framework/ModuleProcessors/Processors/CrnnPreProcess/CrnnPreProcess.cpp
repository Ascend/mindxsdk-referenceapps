#include "CrnnPreProcess.h"
#include "CrnnInferProcess/CrnnInferProcess.h"
#include "Utils.h"

using namespace ascendOCR;

CrnnPreProcess::CrnnPreProcess()
{
    withoutInputQueue_ = false;
    isStop_ = false;
}

CrnnPreProcess::~CrnnPreProcess() {}

APP_ERROR CrnnPreProcess::Init(ConfigParser &configParser, ModuleInitParams &initParams)
{
    LogInfo << "Begin to init instance " << initParams.instanceId;

    InitParams(initParams);
    isClassification = Utils::EndsWith(initParams.pipelineName, "true");

    std::string tempPath("./temp/crnn");
    std::vector<std::string> files;
    Utils::GetAllFiles(tempPath, files);
    for (auto &file : files) {
        std::vector<std::string> nameInfo;
        Utils::StrSplit(file, ".", nameInfo);
        batchSizeList.push_back(uint64_t(std::stoi(nameInfo[nameInfo.size() - 2])));
        if (gearInfo.empty()) {
            Utils::LoadFromFilePair(file, gearInfo);
        }
    }

    std::sort(gearInfo.begin(), gearInfo.end(), Utils::PairCompare);
    std::sort(batchSizeList.begin(), batchSizeList.end(), Utils::UintCompare);
    recMaxWidth = gearInfo[gearInfo.size() - 1].second;
    recMinWidth = gearInfo[0].second;
    mStdHeight = gearInfo[0].first;

    LogInfo << recMinWidth << " " << recMaxWidth << " " << mStdHeight;
    LogInfo << "CrnnPreProcess [" << instanceId_ << "]: Init success.";
    return APP_ERR_OK;
}

APP_ERROR CrnnPreProcess::DeInit(void)
{
    LogInfo << "CrnnPreProcess [" << instanceId_ << "]: Deinit success.";
    return APP_ERR_OK;
}

void CrnnPreProcess::GetGearInfo(int maxResizedW, std::pair<uint64_t, uint64_t> &gear)
{
    if (maxResizedW <= recMaxWidth) {
        auto info = std::upper_bound(gearInfo.begin(), gearInfo.end(),
            std::pair<uint64_t, uint64_t>(mStdHeight, maxResizedW), Utils::GearCompare);
        gear = gearInfo[info - gearInfo.begin()];
    }
}

int CrnnPreProcess::GetCrnnMaxWidth(std::vector<cv::Mat> frames, float maxWHRatio)
{
    int maxResizedW = 0;
    for (auto &frame : frames) {
        int resizedW;
        int imgH = frame.rows;
        int imgW = frame.cols;
        float ratio = imgW / float(imgH);
        int maxWidth = int(maxWHRatio * mStdHeight);
        if (std::ceil(mStdHeight * ratio) > maxWidth) {
            resizedW = maxWidth;
        } else {
            resizedW = int(std::ceil(mStdHeight * ratio));
        }
        maxResizedW = std::max(resizedW, maxResizedW);
        maxResizedW = std::max(std::min(maxResizedW, recMaxWidth), recMinWidth);
    }
    std::pair<uint64_t, uint64_t> gear;
    GetGearInfo(maxResizedW, gear);
    return gear.second;
}
uint8_t *CrnnPreProcess::PreprocessCrnn(std::vector<cv::Mat> &frames, uint32_t BatchSize, int maxResizedW,
    float maxWHRatio, std::vector<ResizedImageInfo> &resizedImageInfos)
{
    cv::Mat resizedImg;
    cv::Mat inImg;
    cv::Mat outImg;
    int resizedW;
    int imgH;
    int imgW;
    uint32_t bufferlen = Utils::RgbImageSizeF32(maxResizedW, mStdHeight);
    uint8_t *srcData = new uint8_t[bufferlen * BatchSize];

    int pos = 0;
    for (uint32_t i = 0; i < frames.size(); i++) {
        inImg = frames[i];
        imgH = inImg.rows;
        imgW = inImg.cols;
        float ratio = imgW / float(imgH);
        int maxWidth = int(maxWHRatio * mStdHeight);
        if (std::ceil(mStdHeight * ratio) > maxWidth) {
            resizedW = maxWidth;
        } else {
            resizedW = int(std::ceil(mStdHeight * ratio));
        }
        resizedW = std::min(resizedW, recMaxWidth);
        cv::resize(inImg, resizedImg, cv::Size(resizedW, mStdHeight));
        int paddingLen = maxResizedW - resizedW;
        if (paddingLen > 0) {
            cv::copyMakeBorder(resizedImg, resizedImg, 0, 0, 0, paddingLen, cv::BORDER_CONSTANT, 0);
        }

        LogDebug << "input image [" << i << "] size / preprocessed image size: " << inImg.size() << "/" <<
            resizedImg.size();

        ResizedImageInfo ResizedInfo;
        ResizedInfo.widthResize = resizedW;
        ResizedInfo.heightResize = mStdHeight;
        ResizedInfo.widthOriginal = inImg.cols;
        ResizedInfo.heightOriginal = inImg.rows;
        resizedImageInfos.emplace_back(std::move(ResizedInfo));

        outImg = resizedImg;
        outImg.convertTo(outImg, CV_32FC3, 1.0 / 255);
        outImg = (outImg - 0.5) / 0.5;

        // Gray channel means
        std::vector<cv::Mat> channels;
        cv::split(outImg, channels);

        // Transform NHWC to NCHW
        uint32_t size = Utils::RgbImageSizeF32(maxResizedW, mStdHeight);
        uint8_t *buffer = Utils::ImageNchw(channels, size);

        // component padding images
        memcpy(srcData + pos, buffer, bufferlen);
        pos += bufferlen;
        delete[] buffer;
    }
    return srcData;
}

std::vector<uint32_t> CrnnPreProcess::GetCrnnBatchSize(uint32_t frameSize)
{
    int lastIndex = batchSizeList.size() - 1;
    std::vector<uint32_t> splitList(frameSize / batchSizeList[lastIndex], batchSizeList[lastIndex]);
    frameSize = frameSize - batchSizeList[lastIndex] * (frameSize / batchSizeList[lastIndex]);
    if (!frameSize) {
        return splitList;
    }
    for (auto bs : batchSizeList) {
        if (frameSize <= bs) {
            splitList.push_back(bs);
            break;
        }
    }
    return splitList;
}

APP_ERROR CrnnPreProcess::Process(std::shared_ptr<void> commonData)
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    uint32_t totalSize = data->imgMatVec.size();
    if (totalSize == 0) {
        data->eof = true;
        auto endTime = std::chrono::high_resolution_clock::now();
        double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        Signal::recPreProcessTime += costTime;
        Signal::e2eProcessTime += costTime;
        SendToNextModule(MT_CrnnInferProcess, data, data->channelId);
        return APP_ERR_OK;
    }

    std::vector<uint32_t> splitIndex = { totalSize };
    splitIndex = GetCrnnBatchSize(totalSize);

    int startIndex = 0;
    int shareId = 0;
    for (size_t i = 0; i < splitIndex.size(); i++) {
        std::shared_ptr<CommonData> dataNew = std::make_shared<CommonData>();

        std::vector<ResizedImageInfo> resizedImageInfosCrnn;
        std::vector<cv::Mat> input(data->imgMatVec.begin() + startIndex,
            data->imgMatVec.begin() + std::min(startIndex + splitIndex[i], totalSize));
        std::vector<std::string> splitRes(data->inferRes.begin() + startIndex,
            data->inferRes.begin() + std::min(startIndex + splitIndex[i], totalSize));
        int maxResizedW = GetCrnnMaxWidth(input, data->maxWHRatio);

        uint8_t *crnnInput = PreprocessCrnn(input, splitIndex[i], maxResizedW, data->maxWHRatio, resizedImageInfosCrnn);
        shareId++;

        dataNew->eof = false;
        dataNew->outputTensorVec = data->outputTensorVec;
        dataNew->imgName = data->imgName;
        dataNew->inferRes = splitRes;
        dataNew->imgTotal = data->imgTotal;
        dataNew->maxResizedW = maxResizedW;

        dataNew->resizedImageInfos = resizedImageInfosCrnn;
        dataNew->batchSize = splitIndex[i];
        dataNew->imgBuffer = crnnInput;
        dataNew->saveFileName = data->saveFileName;
        dataNew->frameSize = std::min(startIndex + splitIndex[i], totalSize) - startIndex;
        dataNew->subImgTotal = data->subImgTotal;
        dataNew->imgId = data->imgId;
        
        startIndex += splitIndex[i];
        SendToNextModule(MT_CrnnInferProcess, dataNew, data->channelId);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    Signal::recPreProcessTime += costTime;
    Signal::e2eProcessTime += costTime;

    return APP_ERR_OK;
}
