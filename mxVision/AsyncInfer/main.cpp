/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: Yolov3+ResNet
 * Author:
 * Create: 2021
 * History: NA
 * */

#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/MxBase.h"
#include <algorithm>
#include <map>
#include <queue>
#include "MxBase/Maths/FastMath.h"
#include "MxBase/postprocess/include/ObjectPostProcessors/Yolov3PostProcess.h"
#include <iostream>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <dirent.h>


using namespace MxBase;
using namespace std;

APP_ERROR ret = APP_ERR_OK;
int32_t g_deviceId = 1;

int BATCH_SIZE = 4;
int STREAM_NUM = 4;

std::vector <MxBase::Image> ImageVec;
ImageProcessor imageProcessor(g_deviceId);

std::vector <AscendStream> AscendStreamVec;
std::vector <AscendStream> StreamVec_;

std::vector <Image> decodeImageVec;
std::vector <Image> resizeImageVec;
std::vector <std::vector<Rect>> cropConfigVecVec;
std::vector <std::vector<Image>> cropResizeImageVecVec;
std::vector <Tensor> tensorImgVec;
std::vector <Tensor> resnetTensorVec;

std::string yoloPath = "./model/yolov3/yolov3_tf_aipp.om";
Model yoloV3(yoloPath, g_deviceId);

std::string resnetPath = "./model/resnet50/resnet50_tensorflow_1.7.om";
Model resnet50(resnetPath, g_deviceId);

std::vector <std::vector<Image>> decodeImageBatch(BATCH_SIZE);
std::vector <std::vector<Image>> resizeImageBatch(BATCH_SIZE);
std::vector <std::vector<Tensor>> tensorImageBatch(BATCH_SIZE);
std::vector <std::vector<Tensor>> resnetTensorBatch(BATCH_SIZE);
std::vector <std::vector<std::vector < Tensor>>>
resnetInputBatch(BATCH_SIZE);
std::vector <std::vector<std::vector < Tensor>>>
yoloV3OutputTensorBatch(BATCH_SIZE);
std::vector <std::vector<std::vector < Tensor>>>
resnetOutputTensorBatch(BATCH_SIZE);
std::vector <std::vector<std::vector < Image>>>
cropResizeImageBatch(BATCH_SIZE);
std::vector <std::vector<std::vector < Rect>>>
cropConfigRectBatch(BATCH_SIZE);
std::vector <std::vector<std::vector < Tensor>>>
yoloV3InputTensorBatch(BATCH_SIZE);

int SplitImage(const std::string &imgDir);

struct Params {
    std::vector <Image> &DecodeImageBatch;
    std::vector <Image> &ResizeImageBatch;
    std::vector <Tensor> &TensorImageBatch;
    std::vector <Tensor> &ResnetTensorBatch;
    std::vector <std::vector<Tensor>> &ResnetInputBatch;
    std::vector <std::vector<Image>> &CropResizeImageBatch;
    std::vector <std::vector<Tensor>> &YoloV3OutputTensorBatch;
    std::vector <std::vector<Tensor>> &ResnetOutputTensorBatch;
    std::vector <std::vector<Rect>> &CropConfigRectBatch;
    std::vector <std::vector<Tensor>> &yoloV3InputTensorBatch;
};

struct AsyncYoloV3PostProcessParam {
    vector <Tensor> &yoloV3Outputs;
    int batchIndex;
    Params *params;
    int dataIndex;
};

struct AsyncResnetYoloV3PostProcessParam {
    vector <Tensor> resnetOutput;
};

struct ConvertToTensorParam {
    bool isYolo;
    Params *params;
    int dataIndex;
};

struct E2eInferParams {
    int batchIdx;
    Params *params;
};

struct MallocYoloTensor {
    Tensor output1;
    Tensor output2;
    Tensor output3;
};

struct MallocResNetTensor {
    Tensor output1;
};

struct HoldResourceParam {
    Tensor outTensor1;
    Tensor outTensor2;
    Tensor outTensor3;
    Tensor resnetOutput1;
    vector <Tensor> *yoloV3Outputs;
    ConvertToTensorParam *convertToTensorParam1;
    ConvertToTensorParam *convertToTensorParam2;
    MallocYoloTensor *mallocYoloTensor;
    MallocResNetTensor *mallocResNetTensor;
    vector <Tensor> *resnetoutput;
    AsyncResnetYoloV3PostProcessParam *asyncResnetYoloV3PostProcessParam;
    AsyncYoloV3PostProcessParam *asyncYoloV3PostProcessParam;
};

void HoldResourceCallback(void *args) {
    HoldResourceParam *input = static_cast<HoldResourceParam * >(args);
    delete input->yoloV3Outputs;
    delete input->convertToTensorParam1;
    delete input->convertToTensorParam2;
    delete input->mallocYoloTensor;
    delete input->mallocResNetTensor;
    delete input->resnetoutput;
    delete input->asyncYoloV3PostProcessParam;
    delete input->asyncResnetYoloV3PostProcessParam;
    delete input;
}

std::vector<E2eInferParams *> E2eInferParamsVec;

void SDKYoloV3PostProcess(std::string &yoloV3ConfigPath, std::string &yoloV3LabelPath, vector <Tensor> &yoloV3Outputs,
                          std::vector <Rect> &cropConfigVec, std::vector <ImagePreProcessInfo> &imagePreProcessInfos) {
    std::map <std::string, std::string> postConfig;
    postConfig.insert(pair<std::string, std::string>("postProcessConfigPath", yoloV3ConfigPath));
    postConfig.insert(pair<std::string, std::string>("labelPath", yoloV3LabelPath));

    Yolov3PostProcess yolov3PostProcess;
    yolov3PostProcess.Init(postConfig);

    std::vector <TensorBase> tensors;
    for (size_t i = 0; i < yoloV3Outputs.size(); i++) {
        MemoryData memoryData(yoloV3Outputs[i].GetData(), yoloV3Outputs[i].GetByteSize());
        TensorBase tensorBase(memoryData, true, yoloV3Outputs[i].GetShape(), TENSOR_DTYPE_INT32);
        tensors.push_back(tensorBase);
    }

    std::vector <std::vector<ObjectInfo>> objectInfos;

    yolov3PostProcess.Process(tensors, objectInfos, imagePreProcessInfos);

    cout << "size of objectInfos is: " << objectInfos.size() << endl;
    for (size_t i = 0; i < objectInfos.size(); i++) {
        cout << "objectInfos-" << i << endl;
        cout << "size of objectInfo-" << i << " is: " << objectInfos[i].size() << endl;
        for (size_t j = 0; j < objectInfos[i].size(); j++) {
            cout << "   objectInfo-" << j << endl;
            cout << "       x0 is: " << objectInfos[i][j].x0 << endl;
            cout << "       y0 is: " << objectInfos[i][j].y0 << endl;
            cout << "       x1 is: " << objectInfos[i][j].x1 << endl;
            cout << "       y1 is: " << objectInfos[i][j].y1 << endl;
            cout << "       confidence is: " << objectInfos[i][j].confidence << endl;
            cout << "       classId is: " << objectInfos[i][j].classId << endl;
            cout << "       className is: " << objectInfos[i][j].className << endl;
        }
    }

    cropConfigVec.resize(objectInfos[0].size());

    for (size_t i = 0; i < objectInfos[0].size(); i++) {
        cropConfigVec[i].x0 = objectInfos[0][i].x0;
        cropConfigVec[i].y0 = objectInfos[0][i].y0;
        cropConfigVec[i].x1 = objectInfos[0][i].x1;
        cropConfigVec[i].y1 = objectInfos[0][i].y1;
    }
}

void ResnetYoloV3PostProcess(vector <Tensor> resnetOutput_) {
    int classNum_ = 1000; // 类别个数

    if (resnetOutput_.empty()) {
        std::cout << "resnet Infer failed.." << std::endl;
        return;
    }

    float *castData = static_cast<float *>(resnetOutput_[0].GetData());
    std::vector<float> result;
    for (int j = 0; j < classNum_; ++j) {
        result.push_back(castData[j]);
    }

    // 计算结果
    std::vector<float>::iterator maxElement = std::max_element(std::begin(result), std::end(result));
    int argmaxIndex = maxElement - std::begin(result);
    float confidence = *maxElement;

    std::cout << "==============推理结果===============" << std::endl;
    cout << "argmaxIndex = " << argmaxIndex << endl;
    cout << "confidence = " << confidence << endl;
}

APP_ERROR PrepareData() {
    string imagePath_ = "./imgs_bak";
    int totalImgs = SplitImage(imagePath_);
    std::vector <std::vector<std::string>> imgFileVecs(BATCH_SIZE);
    for (size_t i = 0; i < imgFileVecs.size(); i++) {
        std::ifstream imgFileStream;
        std::string imagePath = imagePath_ + "/imgSplitFile" + std::to_string(i);
        imgFileStream.open(imagePath);
        std::string imgFile;
        std::vector <std::string> imgVecs;
        while (getline(imgFileStream, imgFile)) {
            imgFileVecs[i].push_back(imgFile);
        }
        imgFileStream.close();
    }
    for (size_t i = 0; i < decodeImageBatch.size(); i++) {
        for (size_t j = 0; j < imgFileVecs[i].size(); j++) {
            Image decodeImage;
            ret = imageProcessor.Decode(imgFileVecs[i][j], decodeImage, ImageFormat::RGB_888);
            if (ret != APP_ERR_OK) {
                std::cout << "imageProcessor Decode failed." << std::endl;
                return -1;
            }
            decodeImageBatch[i].push_back(decodeImage);
        }
    }

    for (int i = 0; i < totalImgs; i++) {
        Image resizedImage;
        resizeImageBatch[i % BATCH_SIZE].push_back(resizedImage);

        Tensor tensorImg;
        tensorImageBatch[i % BATCH_SIZE].push_back(tensorImg);

        Tensor resnetTensor;
        resnetTensorBatch[i % BATCH_SIZE].push_back(resnetTensor);

        vector <Tensor> resnetInput;
        resnetInputBatch[i % BATCH_SIZE].push_back(resnetInput);

        std::vector <Image> cropResizedImageVec;
        cropResizeImageBatch[i % BATCH_SIZE].push_back(cropResizedImageVec);

        vector <Tensor> yoloV3Outputs;
        yoloV3OutputTensorBatch[i % BATCH_SIZE].push_back(yoloV3Outputs);

        vector <Tensor> resnetOutput;
        resnetOutputTensorBatch[i % BATCH_SIZE].push_back(resnetOutput);

        vector <Rect> cropConfigVec;
        cropConfigRectBatch[i % BATCH_SIZE].push_back(cropConfigVec);

        vector <Tensor> yoloV3InputVec;
        yoloV3InputTensorBatch[i % BATCH_SIZE].push_back(yoloV3InputVec);
    }

    for (int i = 0; i < STREAM_NUM; i++) {
        AscendStream Stream_ = AscendStream(g_deviceId);
        Stream_.CreateAscendStream();
        StreamVec_.push_back(Stream_);
    }

    for (int i = 0; i < BATCH_SIZE; i++) {
        AscendStream Stream = AscendStream(g_deviceId);
        Stream.CreateAscendStream();
        AscendStreamVec.push_back(Stream);
    }

    return APP_ERR_OK;
}

APP_ERROR AsyncYoloV3PostProcessPro(vector <Tensor> &yoloV3Outputs, int batchIndex, Params *params, int dataIndex) {
    for (size_t i = 0; i < yoloV3Outputs.size(); i++) {
        ret = yoloV3Outputs[i].ToHost();
    }

    std::cout << "======================yolov3 后处理 =============================" << endl;
    string yoloV3Config = "./model/yolov3/yolov3_tf_bs1_fp16.cfg";
    string yoloV3LabelPath = "./model/yolov3/coco.names";

    ImagePreProcessInfo imagePreProcessInfo(params->ResizeImageBatch[dataIndex].GetOriginalSize().width,
                                            params->ResizeImageBatch[dataIndex].GetOriginalSize().height,
                                            params->DecodeImageBatch[dataIndex].GetOriginalSize().width,
                                            params->DecodeImageBatch[dataIndex].GetOriginalSize().height);
    vector<ImagePreProcessInfo> imagePreProcessInfos{imagePreProcessInfo};

    SDKYoloV3PostProcess(yoloV3Config, yoloV3LabelPath, yoloV3Outputs, params->CropConfigRectBatch[dataIndex],
                         imagePreProcessInfos);
    if (params->CropConfigRectBatch[dataIndex].empty() || params->CropConfigRectBatch[dataIndex].size() == 0) {
        std::cout << "Failed to run yolov3 postProcess." << std::endl;
        return 0;
    }
    params->CropResizeImageBatch[dataIndex].resize(params->CropConfigRectBatch[dataIndex].size());
}

void AsyncYoloV3PostProcessCallbackFunc(void *args) {
    ret = AsyncYoloV3PostProcessPro(static_cast<AsyncYoloV3PostProcessParam *>(args)->yoloV3Outputs,
                                    static_cast<AsyncYoloV3PostProcessParam *>(args)->batchIndex,
                                    static_cast<AsyncYoloV3PostProcessParam *>(args)->params,
                                    static_cast<AsyncYoloV3PostProcessParam *>(args)->dataIndex);
    if (ret != APP_ERR_OK) {
        LogError << "Async execute yolov3 postprocess failed.";
    }
}

APP_ERROR AsyncResNetYoloV3PostPrcessPro(std::vector <Tensor> &resnetOutput_) {
    for (size_t i = 0; i < resnetOutput_.size(); i++) {
        resnetOutput_[i].ToHost();
    }
    ResnetYoloV3PostProcess(resnetOutput_);
    return APP_ERR_OK;
}

void AsyncResNetYoloV3PostProcessCallbackFunc(void *args) {
    AsyncResnetYoloV3PostProcessParam *input = static_cast<AsyncResnetYoloV3PostProcessParam *>(args);
    ret = AsyncResNetYoloV3PostPrcessPro(input->resnetOutput);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Async execute resnet yolov3 postprocess failed.";
    }
}

void ConvertToTensorProcess(void *args) {
    ConvertToTensorParam *convertToTensorParam = static_cast<ConvertToTensorParam * >(args);
    int dataIndex = convertToTensorParam->dataIndex;
    if (convertToTensorParam->isYolo) {
        convertToTensorParam->params->TensorImageBatch[dataIndex] =
                convertToTensorParam->params->ResizeImageBatch[dataIndex].ConvertToTensor();
        convertToTensorParam->params->yoloV3InputTensorBatch[dataIndex] =
                {convertToTensorParam->params->TensorImageBatch[dataIndex]};
    } else {
        convertToTensorParam->params->ResnetTensorBatch[dataIndex] =
                convertToTensorParam->params->CropResizeImageBatch[dataIndex][0].ConvertToTensor();
        convertToTensorParam->params->ResnetInputBatch[dataIndex] =
                {convertToTensorParam->params->ResnetTensorBatch[dataIndex]};
    }
}

void YoloMalloc(void *args) {
    MallocYoloTensor *input = static_cast<MallocYoloTensor * >(args);
    MxBase::Tensor::TensorMalloc(input->output1);
    MxBase::Tensor::TensorMalloc(input->output2);
    MxBase::Tensor::TensorMalloc(input->output3);
}

void ResNetMalloc(void *args) {
    MallocResNetTensor *input = static_cast<MallocResNetTensor * >(args);
    MxBase::Tensor::TensorMalloc(input->output1);
}

APP_ERROR E2eInferAsync(int batchIndex, Params *param) {
    for (int i = 0; i < param->DecodeImageBatch.size(); i++) {
        uint32_t sizeValue1 = 416;
        uint32_t sizeValue2 = 224;
        ret = imageProcessor.Resize(param->DecodeImageBatch[i], Size(sizeValue1, sizeValue1), param->ResizeImageBatch[i],
                                    Interpolation::HUAWEI_HIGH_ORDER_FILTER, AscendStreamVec[batchIndex]);
        if (ret != APP_ERR_OK) {
            std::cout << "imageProcessor Resize failed. ret is " << ret << std::endl;
            return ret;
        }

        ConvertToTensorParam *convertToTensorParam1 = new ConvertToTensorParam {true, param, i};
        ret = AscendStreamVec[batchIndex].LaunchCallBack(ConvertToTensorProcess,
                                                         static_cast<void *>(convertToTensorParam1));

        cout << "====================== 图像前处理结束 =======================" << endl << endl;
        cout << "====================== 目标检测模型推理 ======================" << endl;

        MxBase::Tensor outTensor1({ 1, 13, 13, 255 }, MxBase::TensorDType::FLOAT32, g_deviceId);
        MxBase::Tensor outTensor2({ 1, 26, 26, 255 }, MxBase::TensorDType::FLOAT32, g_deviceId);
        MxBase::Tensor outTensor3( { 1, 52, 52, 255 }, MxBase::TensorDType::FLOAT32, g_deviceId);

        MallocYoloTensor *mallocYoloTensor = new MallocYoloTensor {outTensor1, outTensor2, outTensor3};
        ret = AscendStreamVec[batchIndex].LaunchCallBack(YoloMalloc, static_cast<void *>(mallocYoloTensor));

        vector <Tensor> *yoloV3outputs = new vector <Tensor> {outTensor1, outTensor2, outTensor3};
        ret = yoloV3.Infer(param->yoloV3InputTensorBatch[i], *yoloV3outputs, AscendStreamVec[batchIndex]);

        AsyncYoloV3PostProcessParam *asyncYoloV3PostProcessParam = new AsyncYoloV3PostProcessParam {*yoloV3outputs,
                                                                                                   batchIndex,
                                                                                                   param,
                                                                                                   i};
        ret = AscendStreamVec[batchIndex].LaunchCallBack(AsyncYoloV3PostProcessCallbackFunc,
                                                         static_cast<void * >(asyncYoloV3PostProcessParam));
        ret = imageProcessor.CropResize(param->DecodeImageBatch[i], param->CropConfigRectBatch[i],
                                        Size(sizeValue2, sizeValue2),
                                        param->CropResizeImageBatch[i], AscendStreamVec[batchIndex]);

        ConvertToTensorParam *convertToTensorParam2 = new ConvertToTensorParam {false, param, i};
        ret = AscendStreamVec[batchIndex].LaunchCallBack(ConvertToTensorProcess,
                                                         static_cast<void *>(convertToTensorParam2));

        std::cout << "===================== resnet推理开始 ====================" << std::endl;

        MxBase::Tensor resnetOutput1({1, 1001}, MxBase::TensorDType::FLOAT32, g_deviceId);

        MallocResNetTensor *mallocResNetTensor = new MallocResNetTensor {resnetOutput1};
        ret = AscendStreamVec[batchIndex].LaunchCallBack(ResNetMalloc, static_cast<void *>(mallocResNetTensor));
        vector <Tensor> *resnetoutput = new vector <Tensor> {resnetOutput1};

        resnet50.Infer(param->ResnetInputBatch[i], *resnetoutput, AscendStreamVec[batchIndex]);

        AsyncResnetYoloV3PostProcessParam *asyncResnetYoloV3PostProcessParam = new AsyncResnetYoloV3PostProcessParam {
                *resnetoutput
        };

        ret = AscendStreamVec[batchIndex].LaunchCallBack(AsyncResNetYoloV3PostProcessCallbackFunc,
                                                         static_cast<void *>(asyncResnetYoloV3PostProcessParam));

        HoldResourceParam *holdResourceParam = new HoldResourceParam{outTensor1, outTensor2, outTensor3, resnetOutput1,
                                                                     yoloV3outputs, convertToTensorParam1,
                                                                     convertToTensorParam2, mallocYoloTensor,
                                                                     mallocResNetTensor, resnetoutput,
                                                                     asyncResnetYoloV3PostProcessParam,
                                                                     asyncYoloV3PostProcessParam};
        ret = AscendStreamVec[batchIndex].LaunchCallBack(HoldResourceCallback, static_cast<void *>(holdResourceParam));

    }

    return ret;
}

void AsyncE2eInferProcess(void *args) {
    E2eInferParams *e2EInferParams = static_cast<E2eInferParams * >(args);
    E2eInferAsync(e2EInferParams->batchIdx, e2EInferParams->params);
}

APP_ERROR AsyncE2eInfer(AscendStream &stream, E2eInferParams *e2EInferParams) {
    ret = stream.LaunchCallBack(AsyncE2eInferProcess, static_cast<void * >(e2EInferParams));
    return ret;
}

int SplitImage(const std::string &imgDir) {
    int totalImg = 0;
    DIR *dir = nullptr;
    struct dirent *ptr = nullptr;
    if ((dir = opendir(imgDir.c_str())) == nullptr) {
        LogError << "Open image dir failed, please check the input image dir existed.";
        exit(1);
    }
    int d_type_0 = 10;
    int d_type_1 = 4;
    std::vector <std::string> imgVec;
    while ((ptr = readdir(dir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
            continue;
        } else if (ptr->d_type == d_type_0 || ptr->d_type == d_type_1) {
            continue;
        } else {
            std::string filePath = imgDir + "/" + ptr->d_name;
            imgVec.push_back(filePath);
            totalImg++;
        }
    }
    closedir(dir);
    sort(imgVec.begin(), imgVec.end());

    std::vector <std::vector<std::string>> fileNum(BATCH_SIZE);
    for (size_t i = 0; i < imgVec.size(); i++) {
        fileNum[i % BATCH_SIZE].push_back(imgVec[i]);
    }

    for (int i = 0; i < BATCH_SIZE; i++) {
        std::ofstream imgFile;
        std::string fileName = imgDir + "/imgSplitFile" + std::to_string(i);
        imgFile.open(fileName, ios::out | ios::trunc);
        for (const auto &img: fileNum[i]) {
            imgFile << img << std::endl;
        }
        imgFile.close();
    }
    LogInfo << "Split Image success.";
    return totalImg;
}

int main(int argc, char *argv[]) {
    MxInit();

    PrepareData();

    for (int i = 0; i < BATCH_SIZE; i++) {
        Params *params = new Params{decodeImageBatch[i], resizeImageBatch[i], tensorImageBatch[i], resnetTensorBatch[i],
                                    resnetInputBatch[i], cropResizeImageBatch[i], yoloV3OutputTensorBatch[i],
                                    resnetOutputTensorBatch[i], cropConfigRectBatch[i], yoloV3InputTensorBatch[i]};
        E2eInferParams *e2EInferParams = new E2eInferParams{i, params};
        E2eInferParamsVec.push_back(e2EInferParams);
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BATCH_SIZE; i++) {
        ret = AsyncE2eInfer(StreamVec_[i % STREAM_NUM], E2eInferParamsVec[i]);
    }
    for (int i = 0; i < STREAM_NUM; i++) {
        StreamVec_[i].Synchronize();
    }
    for (int i = 0; i < BATCH_SIZE; i++) {
        AscendStreamVec[i].Synchronize();
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    cout << "all time is " << costTime << "ms" << endl << endl;
    if (ret != APP_ERR_OK) {
        LogError << "Failed to run E2eInfer.";
    }

    for (int i = 0; i < STREAM_NUM; i++) {
        StreamVec_[i].DestroyAscendStream();
    }
    for (int i = 0; i < BATCH_SIZE; i++) {
        AscendStreamVec[i].DestroyAscendStream();
    }

    for (int i = 0; i < BATCH_SIZE; i++) {
        std::string filePath = "./imgs_bak/imgSplitFile" + std::to_string(i);
        auto ret = remove(filePath.c_str());
        if (ret != 0) {
            std::cout << "remove file [" << filePath << "] failed." << endl;
            return -1;
        } else {
            std::cout << "remove file [" << filePath << "] failed." << endl;
        }
    }
    return 0;
}
