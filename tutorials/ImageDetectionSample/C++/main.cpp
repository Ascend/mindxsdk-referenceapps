#include <string>
#include <glob.h>
#include "opencv2/opencv.hpp"
#include "MxBase/Log/Log.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxStream/StreamManager/MxStreamManager.h"

namespace {
    const uint32_t YUV_BYTES_NU = 3;
    const uint32_t YUV_BYTES_DE = 2;
}

// 读取文件中的信息
static APP_ERROR  ReadFile(const std::string& filePath, MxStream::MxstDataInput& dataBuffer)
{
    char c[PATH_MAX + 1] = {0x00};
    size_t count = filePath.copy(c, PATH_MAX +1);
    if(count != filePath.length()){
        LogError << "Failed to copy file path(" << c << ").";
        return APP_ERR_COMM_FAILURE;
    }
    // 得到文件的绝对路径
    char path[PATH_MAX + 1] = { 0x00 };
    if((strlen(c) > PATH_MAX) || (realpath(c, path) == nullptr)){
        LogError << "Failed to get image, the image path is (" << filePath << ").";
        return APP_ERR_COMM_NO_EXIST;
    }
    // 打开文件
    // path里面的值是test.jpg文件的绝对路径
    FILE *fp = fopen(path, "rb");
    if(fp == nullptr){
        LogError << "Failed to open file (" << path << ").";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // 得到文件内容长度
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // 若文件内容不为空，把文件内容写入dataBuffer中
    if(fileSize > 0){
        dataBuffer.dataSize = fileSize;
        dataBuffer.dataPtr = new (std::nothrow) uint32_t[fileSize];
        if(dataBuffer.dataPtr == nullptr){
            LogError << "allocate memory with \"new uint32_t\" failed.";
            fclose(fp);
            return APP_ERR_COMM_FAILURE;
        }

        uint32_t readRet = fread(dataBuffer.dataPtr, 1, fileSize, fp);
        if(readRet <= 0){
            fclose(fp);
            return APP_ERR_COMM_READ_FAIL;
        }
        fclose(fp);
        return APP_ERR_OK;
    }
    fclose(fp);
    return APP_ERR_COMM_FAILURE;
}

// 读取pipeline信息
static std::string ReadPipelineConfig(const std::string &pipelineConfigPath)
{
    // 用二进制方式打开文件
    std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
    if(!file){
        LogError << pipelineConfigPath << " file is not exists";
        return "";
    }

    // 得到文件大小
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();

    file.seekg(0);
    std::unique_ptr<char[]> data(new char[fileSize]);

    // 将文件中读取的信息存入data中
    file.read(data.get(), fileSize);
    file.close();
    // 将读取到的文件内容存储到字符串中并return出去
    std::string pipelineConfig(data.get(), fileSize);
    return pipelineConfig;
}

// 结果可视化
static APP_ERROR SaveResult(const std::shared_ptr<MxTools::MxpiVisionList> &mxpiVisionList,
                            const std::shared_ptr<MxTools::MxpiObjectList> &mxpiObjectList)
{
    // 处理输出原件的protobuf结果信息
    auto& visionInfo = mxpiVisionList->visionvec(0).visioninfo();
    auto& visionData = mxpiVisionList->visionvec(0).visiondata();
    MxBase::MemoryData memorySrc = {};
    memorySrc.deviceId = visionData.deviceid();
    memorySrc.type = (MxBase::MemoryData::MemoryType) visionData.memtype();
    memorySrc.size = visionData.datasize();
    memorySrc.ptrData = (void*)visionData.dataptr();
    MxBase::MemoryData memoryDst(visionData.datasize(), MxBase::MemoryData::MEMORY_HOST_NEW);
    APP_ERROR  ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, memorySrc);
    if(ret != APP_ERR_OK){
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }
    // 用输出原件信息初始化OpenCV图像信息矩阵
    cv::Mat imgYuv = cv::Mat(visionInfo.heightaligned() * YUV_BYTES_NU / YUV_BYTES_DE,
                             visionInfo.widthaligned(), CV_8UC1, memoryDst.ptrData);
    cv::Mat imgBgr = cv::Mat(visionInfo.heightaligned(), visionInfo.widthaligned(), CV_8UC3);
    // 颜色空间转换
    cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV12);

    // 设置OpenCV中的颜色
    const cv::Scalar green = cv::Scalar(0, 255, 0);
    const uint32_t thickness = 4;
    const uint32_t xOffset = 10;
    const uint32_t yOffset = 10;
    const uint32_t lineType = 8;
    const float fontScale = 1.0;
    for(uint32_t i = 0; i < (uint32_t)mxpiObjectList->objectvec_size(); i++){
        auto& object = mxpiObjectList->objectvec(i);
        uint32_t y0 = object.y0();
        uint32_t x0 = object.x0();
        uint32_t y1 = object.y1();
        uint32_t x1 = object.x1();
        // 在图像上绘制文字
        cv::putText(imgBgr, object.classvec(0).classname(), cv::Point(x0 + xOffset, y0 + yOffset),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, green, thickness, lineType);
        // 绘制矩形
        cv::rectangle(imgBgr,cv::Rect(x0, y0, x1 - x0, y1 - y0),
                      green, thickness);
    }
    // 把Mat类型的图像矩阵保存为图像到指定位置。
    cv::imwrite("./result.jpg", imgBgr);
    ret = MxBase::MemoryHelper::MxbsFree(memoryDst);
    if(ret != APP_ERR_OK){
        LogError << "Fail to MxbsFree memory.";
        return ret;
    }
    return APP_ERR_OK;
}

// 打印protobuf信息
static APP_ERROR  PrintInfo(std::vector<MxStream::MxstProtobufOut> outPutInfo)
{
    if(outPutInfo.size() == 0){
        LogError << "outPutInfo size is 0";
        return APP_ERR_ACL_FAILURE;
    }
    if(outPutInfo[0].errorCode != APP_ERR_OK){
        LogError << "GetProtobuf error. errorCode=" << outPutInfo[0].errorCode;
        return outPutInfo[0].errorCode;
    }

    for(MxStream::MxstProtobufOut info : outPutInfo){
        LogInfo << "errorCode=" << info.errorCode;
        LogInfo << "key=" << info.messageName;
        LogInfo << "value=" << info.messagePtr.get()->DebugString();
    }

    return APP_ERR_OK;
}

int main(int argc, char* argv[])
{
    // 读取test.pipeline文件信息
    std::string pipelineConfigPath = "./test.pipeline";
    std::string pipelineConfig = ReadPipelineConfig(pipelineConfigPath);
    if(pipelineConfig == ""){
        return APP_ERR_COMM_INIT_FAIL;
    }

    std::string streamName = "detection";
    // 新建一个流管理MxStreamManager对象并初始化
    auto mxStreamManager = std::make_shared<MxStream::MxStreamManager>();
    APP_ERROR ret = mxStreamManager->InitManager();
    if(ret != APP_ERR_OK){
        LogError << GetError(ret) << "Fail to init Stream manager.";
        return ret;
    }
    // 加载pipeline得到的信息，创建一个新的stream业务流
    ret = mxStreamManager->CreateMultipleStreams(pipelineConfig);
    if(ret != APP_ERR_OK){
        LogError << GetError(ret) << "Fail to creat Stream.";
        return ret;
    }

    // 将图片的信息读取到dataBuffer中
    MxStream::MxstDataInput dataBuffer;
    ret = ReadFile("./test.jpg", dataBuffer);
    if(ret != APP_ERR_OK){
        LogError << "Fail to read image file, ret = " << ret << ".";
        return ret;
    }
    // 通过SendData函数传递输入信息到指定的工作元件模块
    // streamName是pipeline文件中业务流名称；inPluginId为输入端口编号，对应输入元件的编号
    ret = mxStreamManager->SendData(streamName, 0, dataBuffer);
    if(ret != APP_ERR_OK){
        delete dataBuffer.dataPtr;
        LogError << "Fail to send data to stream, ret = " << ret << ".";
        return ret;
    }
    delete dataBuffer.dataPtr;
    // 获得Stream上输出原件的protobuf结果
    std::vector<std::string> keyVec = {"mxpi_objectpostprocessor0", "mxpi_imagedecoder0"};
    std::vector<MxStream::MxstProtobufOut> output = mxStreamManager->GetProtobuf(streamName, 0, keyVec);
    ret = PrintInfo(output);
    if(ret != APP_ERR_OK){
        LogError << "Fail to print the info of output, ret = " << ret << ".";
        return ret;
    }

    // mxpi_objectpostprocessor0模型后处理插件输出信息
    auto objectList = std::static_pointer_cast<MxTools::MxpiObjectList>(output[0].messagePtr);
    // mxpi_imagedecoder0图片解码插件输出信息
    auto mxpiVision = std::static_pointer_cast<MxTools::MxpiVisionList>(output[1].messagePtr);
    // 将结果写入本地图片中
    SaveResult(mxpiVision, objectList);
    mxStreamManager->DestroyAllStreams();
    return 0;
}








