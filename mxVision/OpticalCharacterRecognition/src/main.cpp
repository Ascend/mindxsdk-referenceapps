/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Main file.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include "Utils.h"
#include "Signal.h"
#include "HandOutProcess/HandOutProcess.h"
#include "DbnetPreProcess/DbnetPreProcess.h"
#include "DbnetInferProcess/DbnetInferProcess.h"
#include "DbnetPostProcess/DbnetPostProcess.h"
#include "ClsPreProcess/ClsPreProcess.h"
#include "ClsInferProcess/ClsInferProcess.h"
#include "ClsPostProcess/ClsPostProcess.h"
#include "CrnnPreProcess/CrnnPreProcess.h"
#include "CrnnInferProcess/CrnnInferProcess.h"
#include "CrnnPostProcess/CrnnPostProcess.h"
#include "CollectProcess/CollectProcess.h"

#include "ConfigParser/ConfigParser.h"
#include "ArgumentParser/ArgumentParser.h"
#include "ModuleManagers/ModuleManager.h"
#include "Log/Log.h"

#include "MxBase/MxBase.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <experimental/filesystem>
#include <csignal>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <dirent.h>

using namespace ascendOCR;
using namespace std;

namespace {
void SigHandler(int signal)
{
    if (signal == SIGINT) {
        Signal::signalRecieved = true;
    }
}

void ModuleDescGenerator(int device_num, std::vector<ascendOCR::ModuleDesc> &moduleDesc, bool isClassification)
{
    moduleDesc.push_back({ MT_HandOutProcess, 1 });
    moduleDesc.push_back({ MT_DbnetPreProcess, static_cast<int>(std::ceil(0.6 * device_num)) });
    moduleDesc.push_back({ MT_DbnetInferProcess, static_cast<int>(std::ceil(1 * device_num)) });
    moduleDesc.push_back({ MT_DbnetPostProcess, static_cast<int>(std::ceil(2 * device_num)) });

    // Classification
    if (isClassification) {
        moduleDesc.push_back({ MT_ClsPreProcess, static_cast<int>(std::ceil(0.6 * device_num)) });
        moduleDesc.push_back({ MT_ClsInferProcess, static_cast<int>(std::ceil(1 * device_num)) });
        moduleDesc.push_back({ MT_ClsPostProcess, static_cast<int>(std::ceil(0.6 * device_num)) });
    }

    // Character Recognition
    moduleDesc.push_back({ MT_CrnnPreProcess, static_cast<int>(std::ceil(0.7 * device_num)) });
    moduleDesc.push_back({ MT_CrnnInferProcess, static_cast<int>(std::ceil(1 * device_num)) });
    moduleDesc.push_back({ MT_CrnnPostProcess, static_cast<int>(std::ceil(0.26 * device_num)) });

    moduleDesc.push_back({ MT_CollectProcess, 1 });
}

void ModuleConnectDesc(std::vector<ascendOCR::ModuleConnectDesc> &connectDesc, bool isClassification)
{
    connectDesc.push_back({ MT_HandOutProcess, MT_DbnetPreProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_DbnetPreProcess, MT_DbnetInferProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_DbnetInferProcess, MT_DbnetPostProcess, MODULE_CONNECT_RANDOM });
    std::string preModule;

    if (isClassification) {
        connectDesc.push_back({ MT_DbnetPostProcess, MT_ClsPreProcess, MODULE_CONNECT_RANDOM });
        connectDesc.push_back({ MT_ClsPreProcess, MT_ClsInferProcess, MODULE_CONNECT_RANDOM });
        connectDesc.push_back({ MT_ClsInferProcess, MT_ClsPostProcess, MODULE_CONNECT_RANDOM });
        preModule = MT_ClsPostProcess;
    } else {
        preModule = MT_DbnetPostProcess;
    }

    connectDesc.push_back({ preModule, MT_CrnnPreProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_CrnnPreProcess, MT_CrnnInferProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_CrnnInferProcess, MT_CrnnPostProcess, MODULE_CONNECT_RANDOM });
    connectDesc.push_back({ MT_CrnnPostProcess, MT_CollectProcess, MODULE_CONNECT_RANDOM });
}

void DescGenerator(std::string &configPath, std::vector<ascendOCR::ModuleConnectDesc> &connectDesc,
    std::vector<ascendOCR::ModuleDesc> &moduleDesc, bool isClassification)
{
    ConfigParser config;
    config.ParseConfig(configPath);
    std::vector<uint32_t> deviceIdVec;
    APP_ERROR ret = config.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret != APP_ERR_OK) {
        LogError << "Get Device ID failed.";
        exit(-1);
    }
    int device_num = (int)deviceIdVec.size();

    ModuleDescGenerator(device_num, moduleDesc, isClassification);

    ModuleConnectDesc(connectDesc, isClassification);
}

APP_ERROR InitModuleManager(ModuleManager &moduleManager, std::string &configPath, std::string &aclConfigPath,
    const std::string &pipeline, bool isClassification)
{
    std::vector<ascendOCR::ModuleConnectDesc> connectDesc;
    std::vector<ascendOCR::ModuleDesc> moduleDesc;
    DescGenerator(configPath, connectDesc, moduleDesc, isClassification);

    LogInfo << "ModuleManager: begin to init.";
    APP_ERROR ret = moduleManager.Init(configPath, aclConfigPath);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to init system manager, ret = " << ret;
        return APP_ERR_COMM_FAILURE;
    }

    ret = moduleManager.RegisterModules(pipeline, moduleDesc.data(), (int)moduleDesc.size(), 0);

    if (ret != APP_ERR_OK) {
        return APP_ERR_COMM_FAILURE;
    }

    ret = moduleManager.RegisterModuleConnects(pipeline, connectDesc.data(), (int)connectDesc.size());

    if (ret != APP_ERR_OK) {
        LogError << "Fail to connect module, ret = " << ret;
        return APP_ERR_COMM_FAILURE;
    }

    return APP_ERR_OK;
}

APP_ERROR DeInitModuleManager(ModuleManager &moduleManager)
{
    APP_ERROR ret = moduleManager.DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "Fail to deinit system manager, ret = " << ret;
        return APP_ERR_COMM_FAILURE;
    }

    return APP_ERR_OK;
}

inline void MainAssert(int exp)
{
    if (exp != APP_ERR_OK) {
        exit(exp);
    }
}

void MainProcess(const std::string &streamName, std::string config, bool isClassification)
{
    LogInfo << "streamName: "<< streamName;
    std::string aclConfig;

    std::chrono::high_resolution_clock::time_point endTime;
    std::chrono::high_resolution_clock::time_point startTime;

    ModuleManager moduleManager;
    try {
        MainAssert(InitModuleManager(moduleManager, config, aclConfig, streamName, isClassification));
    } catch (...) {
        LogError << "error occurred during init module.";
        return;
    }

    startTime = std::chrono::high_resolution_clock::now();
    try {
        MainAssert(moduleManager.RunPipeline());
    } catch (...) {
        LogError << "error occurred during start pipeline.";
        return;
    }

    LogInfo << "wait for exit signal";
    if (signal(SIGINT, SigHandler) == SIG_ERR) {
        LogInfo << "cannot catch SIGINT.";
    }
    const uint16_t signalCheckInterval = 1000;
    while (!Signal::signalRecieved) {
        usleep(signalCheckInterval);
    }
    endTime = std::chrono::high_resolution_clock::now();

    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    try {
        MainAssert(DeInitModuleManager(moduleManager));
    } catch (...) {
        LogError << "error occurred during deinit module manager.";
        return;
    }

    LogInfo << "DeInitModuleManager: " << streamName;

    LogInfo << "end to end cost: " << costMs << " in ms / " << costMs / 1000 << " in s. ";
}

int SplitImage(int threadNum, const std::string &imgDir, bool isClassification)
{
    int totalImg = 0;
    DIR *dir = nullptr;
    struct dirent *ptr = nullptr;
    if ((dir = opendir(imgDir.c_str())) == nullptr) {
        LogError << "Open image dir failed, please check the input image dir existed.";
        exit(1);
    }
    std::vector<std::string> imgVec;
    while ((ptr = readdir(dir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
            continue;
        } else if (ptr->d_type == 10 || ptr->d_type == 4) {
            continue;
        } else {
            std::string filePath = imgDir + "/" + ptr->d_name;
            imgVec.push_back(filePath);
            totalImg++;
        };
    }
    closedir(dir);
    sort(imgVec.begin(), imgVec.end());

    std::vector<std::vector<std::string>> fileNum(threadNum);
    for (size_t i = 0; i < imgVec.size(); i++) {
        fileNum[i % threadNum].push_back(imgVec[i]);
    }

    for (int i = 0; i < threadNum; i++) {
        std::ofstream imgFile;
        std::string fileName = "./data/config/imgSplitFile" + std::to_string(i) + Utils::BoolCast(isClassification);
        imgFile.open(fileName, ios::out | ios::trunc);
        for (const auto &img : fileNum[i]) {
            imgFile << img << std::endl;
        }
        imgFile.close();
    }
    LogInfo << "Split Image success.";
    return totalImg;
}

APP_ERROR ParseCommandArgs(int argc, const char *argv[], ArgumentParser &argumentParser)
{
    LogDebug << "Begin to parse and check command arguments.";
    argumentParser.AddArgument("-image_path", "./data/imagePath", "The path of input images, default: ./data/imagePath");
    argumentParser.AddArgument("-thread_num", "1", "The number of threads for the program, default: 1");
    argumentParser.AddArgument("-direction_classification", "false", "perform text direction classification "
                                                                     "using cls model, default: false");
    argumentParser.AddArgument("-config", "./data/config/setup.config", "The path of config file.");
    argumentParser.ParseArgs(argc, argv);

    std::string inputImagePath = argumentParser.GetStringArgumentValue("-image_path");
    APP_ERROR ret = Utils::CheckPath(inputImagePath, "Input images path");
    if (ret != APP_ERR_OK) {
        LogError << "Parse the path of input images failed, please check if the path is correct.";
        return ret;
    }

    int threadNum = argumentParser.GetIntArgumentValue("-thread_num");
    if (threadNum < 1) {
        LogError << "The number of threads cannot be less than one.";
        return APP_ERR_COMM_FAILURE;
    }

    std::string configFilePath = argumentParser.GetStringArgumentValue("-config");
    ret = Utils::CheckPath(configFilePath, "Config file path");
    if (ret != APP_ERR_OK) {
        LogError << "Parse the path of config file failed, please check if the path is correct.";
        return ret;
    }

    return APP_ERR_OK;
}


void saveModelGear(std::string modelPath, int32_t &deviceId, const std::string &modelType)
{
    MxBase::Model model(modelPath, deviceId);
    std::vector<std::vector<uint64_t>> dynamicGearInfo = model.GetDynamicGearInfo();
    std::vector<std::pair<uint64_t, uint64_t>> gearInfo;
    uint64_t batchInfo;
    for (auto &info : dynamicGearInfo) {
        gearInfo.emplace_back(info[2], info[3]);
        batchInfo = info[0];
    }

    std::string baseName = Utils::BaseName(modelPath) + "." + std::to_string(batchInfo) + ".bin";
    std::string savePath = "./temp/" + modelType + "/";
    Utils::MakeDir(savePath, false);

    Utils::SaveToFilePair(savePath + baseName, gearInfo);
}

void saveModelBs(std::string modelPath, int32_t &deviceId, const std::string &modelType)
{
    MxBase::Model model(modelPath, deviceId);
    std::vector<std::vector<uint64_t>> dynamicGearInfo = model.GetDynamicGearInfo();
    std::vector<uint64_t> batchInfo;
    for (auto &info : dynamicGearInfo) {
        batchInfo.emplace_back(info[0]);
    }

    std::string baseName = Utils::BaseName(modelPath) + ".bin";
    std::string savePath = "./temp/" + modelType + "/";
    Utils::MakeDir(savePath, false);

    Utils::SaveToFileVec(savePath + baseName, batchInfo);
}

APP_ERROR configGenerate(std::string &configPath, bool isClassification)
{
    std::string modelConfigPath("./temp");
    Utils::MakeDir(modelConfigPath, true);
    ConfigParser config;
    config.ParseConfig(configPath);

    std::vector<uint32_t> deviceIdVec;
    APP_ERROR ret = config.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret!= APP_ERR_OK || deviceIdVec.empty()) {
        LogError << "Get Device ID failed.";
        exit(-1);
    }
    int32_t deviceId_ = deviceIdVec[0];

    std::string detModelPath;
    ret = config.GetStringValue("detModelPath", detModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Parse the config file path failed, please check if the path is correct.";
        return ret;
    }

    saveModelGear(detModelPath, deviceId_, "dbnet");
    if (isClassification) {
        std::string clsModelPath;
        ret = config.GetStringValue("clsModelPath", clsModelPath);
        if (ret != APP_ERR_OK) {
            LogError << "Parse the config file path failed, please check if the path is correct.";
            return ret;
        }
        saveModelBs(clsModelPath, deviceId_, "cls");
    }

    std::string recModelPath;
    ret = config.GetStringValue("recModelPath", recModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Get recModelPath failed, please check the value of recModelPath";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::vector<std::string> files;
    Utils::GetAllFiles(recModelPath, files);
    for (const auto &file : files) {
        if (Utils::EndsWith(file, ".om")) {
            saveModelGear(file, deviceId_, "crnn");
        }
    }

    return APP_ERR_OK;
}
}


APP_ERROR args_check(const std::string &configPath, bool isClassification)
{
    ConfigParser configParser;
    configParser.ParseConfig(configPath);
    std::string model_path;

    // device id check
    std::vector<uint32_t> deviceIdVec;
    APP_ERROR ret = configParser.GetVectorUint32Value("deviceId", deviceIdVec);
    if (ret != APP_ERR_OK || deviceIdVec.empty()) {
        LogError << "Get device id failed, please check the value of deviceId";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    int32_t deviceId;
    for (auto &deviceId_ : deviceIdVec) {
        deviceId = (int32_t)deviceId_;
        if (deviceId_ < 0 || deviceId_ > 7) {
            LogError << "deviceId must between [0,7]";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }
    // device type check
    std::string deviceType;
    ret = configParser.GetStringValue("deviceType", deviceType);
    if (ret != APP_ERR_OK) {
        LogError << "Get device type failed, please check the value of deviceType";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (deviceType != "310P") {
        LogError << "Device type only support 310P, please check the value of device type.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // det model check
    std::string detModelPath;
    ret = configParser.GetStringValue("detModelPath", detModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Get detModelPath failed, please check the value of detModelPath.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    ret = Utils::CheckPath(detModelPath, "detModelPath");
    if (ret!= APP_ERR_OK) {
        LogError << "detModelPath: " << detModelPath << "is not exist or can not read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    try {
        MxBase::Model model(detModelPath, deviceId);
        std::vector<std::vector<uint64_t>> dynamicGearInfo = model.GetDynamicGearInfo();
        if (dynamicGearInfo.empty()) {
            LogError << "please check the value of detModelPath";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    } catch (...) {
        LogError << "please check the value of related parameters.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    if (isClassification) {
        ret = configParser.GetStringValue("clsModelPath", model_path);
        if (ret != APP_ERR_OK) {
            LogError << "Parse the config file path failed, please check if the path is correct.";
            return ret;
        }

        try {
            MxBase::Model model(model_path, deviceId);
            std::vector<std::vector<uint64_t>> dynamicGearInfo = model.GetDynamicGearInfo();
            LogError << "Cls: ";
            for (auto &info : dynamicGearInfo) {
                LogError << info[2] << " ---- " << info[3];
            }
            if (dynamicGearInfo.empty()) {
                LogError << "please check the value of clsModelPath.";
                return APP_ERR_COMM_INVALID_PARAM;
            }
        } catch (...) {
            LogError << "please check the value of related parameters.";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }
    std::string recModelPath;
    ret = configParser.GetStringValue("recModelPath", recModelPath);
    if (ret != APP_ERR_OK) {
        LogError << "Get recModelPath failed, please check the value of recModelPath.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    ret = Utils::CheckPath(recModelPath, "recModelPath");
    if (ret != APP_ERR_OK) {
        LogError << "rec model path: " << recModelPath << " is not exist of can not read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::vector<std::string> files;
    Utils::GetAllFiles(recModelPath, files);
    for (auto &file : files) {
        try {
            MxBase::Model crnn(file, deviceId);
            std::vector<std::vector<uint64_t>> dynamicGearInfo = crnn.GetDynamicGearInfo();
            if (dynamicGearInfo.empty()) {
                LogError << "please check the value of recModelPath.";
                return APP_ERR_COMM_INVALID_PARAM;
            }
        } catch (...) {
            LogError << "please check the value of related parameters.";
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }

    std::string recDictionary;
    ret = configParser.GetStringValue("dictPath", recDictionary);
    if (ret != APP_ERR_OK) {
        LogError << "Get dictPath failed, please check the value of dictPath.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    ret = Utils::CheckPath(recDictionary, "character label file");
    if (ret != APP_ERR_OK) {
        LogError << "Character label file: " << recDictionary << " does not exist or cannot be read.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    bool saveInferResult;
    ret = configParser.GetBoolValue("saveInferResult", saveInferResult);
    if (ret != APP_ERR_OK) {
        LogError << "Get saveInferResult failed, please check the value of saveInferResult.";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    return APP_ERR_OK;
}

int main(int argc, const char *argv[])
{
    // Initialize
    MxBase::MxInit();

    // Argument Parser
    ArgumentParser argumentParser;
    APP_ERROR ret = ParseCommandArgs(argc, argv, argumentParser);
    if (ret != APP_ERR_OK) {
        LogError << "Parse command arguments failed.";
        exit(-1);
    }
    int threadNum = argumentParser.GetIntArgumentValue("-thread_num");
    std::string inputImagePath = argumentParser.GetStringArgumentValue("-image_path");
    bool isDirectionClassification = argumentParser.GetBoolArgumentValue("-direction_classification");
    int ImageNum = SplitImage(threadNum, inputImagePath, isDirectionClassification);

    // Parameter Check
    if (threadNum > ImageNum) {
        LogError << "thread number [" << threadNum << "] can not bigger than total number of input images [" <<
            ImageNum << "].";
        exit(-1);
    }

    if (threadNum < 1) {
        LogError << "thread number [" << threadNum << "] cannot be smaller than 1.";
        exit(-1);
    }

    if (threadNum > 4) {
        LogError << "thread number [" << threadNum << "] cannot be great than 4.";
        exit(-1);
    }

    Signal::GetInstance().SetThreadNum(threadNum);

    std::string setupConfig = argumentParser.GetStringArgumentValue("-config");
    MainAssert(args_check(setupConfig, isDirectionClassification));

    ret = configGenerate(setupConfig, isDirectionClassification);
    if (ret != APP_ERR_OK) {
        LogError << "config set up failed.";
        exit(-1);
    }

    std::thread threadProcess[threadNum];
    std::string streamName[threadNum];

    for (int i = 0; i < threadNum; ++i) {
        streamName[i] = "imgSplitFile" + std::to_string(i) + Utils::BoolCast(isDirectionClassification);
        threadProcess[i] = std::thread(MainProcess, streamName[i], setupConfig, isDirectionClassification);
    }

    for (int j = 0; j < threadNum; ++j) {
        threadProcess[j].join();
    }

    std::string modelConfigPath("./temp");
    if (access(modelConfigPath.c_str(), 0) != -1) {
        system(("rm -r " + modelConfigPath).c_str());
        LogInfo << modelConfigPath << " removed!";
    }
    LogInfo << "MxOCR Average Process Time: " << Signal::e2eProcessTime / ImageNum << "ms.";
    LogInfo << "program End.";
    return 0;
}
