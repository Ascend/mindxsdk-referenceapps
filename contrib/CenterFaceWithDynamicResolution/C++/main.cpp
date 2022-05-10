/*
 * Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.
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
#include "MxBase/Log/Log.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "opencv2/opencv.hpp"
#include <MxBase/PostProcessBases/PostProcessDataType.h>
#include <glob.h>
#include <string>
#include <sys/stat.h>
namespace {
const uint32_t YUV_BYTES_NU = 3;
const uint32_t YUV_BYTES_DE = 2;
} // namespace

// Read the information in the file
static APP_ERROR  readfile(const std::string& filePath, MxStream::MxstDataInput& dataBuffer)
{
    char c[PATH_MAX] = {0x00};
    size_t count = filePath.copy(c, PATH_MAX);
    if (count != filePath.length()) {
        LogError << "Failed to copy file path(" << c << ").";
        return APP_ERR_COMM_FAILURE;
    }
    // Gets the absolute path to the file
    char path[PATH_MAX] = { 0x00 };
    if ((strlen(c) > PATH_MAX) || (realpath(c, path) == nullptr)) {
        LogError << "Failed to get image, the image path is (" << filePath << ").";
        return APP_ERR_COMM_NO_EXIST;
    }
    // Open the file
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LogError << "Failed to open file (" << path << ").";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // Gets the length of the file content
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // If the contents of the file are not empty, write the contents of the file to dataBuffer
    if (fileSize > 0) {
        dataBuffer.dataSize = fileSize;
        dataBuffer.dataPtr = new (std::nothrow) uint32_t[fileSize]; // Memory is allocated based on file length
        if (dataBuffer.dataPtr == nullptr) {
            LogError << "allocate memory with \"new uint32_t\" failed.";
            fclose(fp);
            return APP_ERR_COMM_FAILURE;
        }
        uint32_t readRet = fread(dataBuffer.dataPtr, 1, fileSize, fp);
        if (readRet <= 0) {
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
static std::string ReadPipelineConfig(const std::string &pipelineConfigPath) {
  // 用二进制方式打开文件
  std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
  if (!file) {
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

static std::string ResolvePathName(const std::string &filepath) {
  size_t npos = filepath.rfind('/');
  if (npos == std::string::npos)
    return std::string();
  return filepath.substr(0, npos);
}
// 创建文件夹
static bool MkdirRecursive(const std::string &filepath) {
  int mode = 0777;
  int ret = mkdir(filepath.c_str(), mode);
  if (ret == 0 || errno == EEXIST) {
    return true;
  } else {
    std::string parent = ResolvePathName(filepath);
    if (parent.empty() || !MkdirRecursive(parent))
      return false;
    return mkdir(filepath.c_str(), mode) == 0;
  }
}

// 结果可视化
static APP_ERROR SaveResult(const std::shared_ptr<MxTools::MxpiVisionList> &mxpiVisionList,
                            const std::shared_ptr<MxTools::MxpiObjectList> &mxpiObjectList,
                            const std::shared_ptr<MxTools::MxpiPoseList> &keypointList) {
  // 处理输出原件的protobuf结果信息
  auto &visionInfo = mxpiVisionList->visionvec(0).visioninfo();
  auto &visionData = mxpiVisionList->visionvec(0).visiondata();
  MxBase::MemoryData memorySrc = {};
  memorySrc.deviceId = visionData.deviceid();
  memorySrc.type = (MxBase::MemoryData::MemoryType)visionData.memtype();
  memorySrc.size = visionData.datasize();
  memorySrc.ptrData = (void *)visionData.dataptr();
  MxBase::MemoryData memoryDst(visionData.datasize(),
                               MxBase::MemoryData::MEMORY_HOST_NEW);
  APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, memorySrc);
  if (ret != APP_ERR_OK) {
    LogError << "Fail to malloc and copy host memory.";
    return ret;
  }
  // 用输出原件信息初始化OpenCV图像信息矩阵
  cv::Mat imgYuv =
      cv::Mat(visionInfo.heightaligned() * YUV_BYTES_NU / YUV_BYTES_DE,
              visionInfo.widthaligned(), CV_8UC1, memoryDst.ptrData);
  cv::Mat imgBgr =
      cv::Mat(visionInfo.heightaligned(), visionInfo.widthaligned(), CV_8UC3);
  // 颜色空间转换
  cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR_NV12);

  // 设置OpenCV中的颜色
  const cv::Scalar green = cv::Scalar(127, 255, 0);
  const cv::Scalar blue = cv::Scalar(255, 0, 0);
  const cv::Scalar red = cv::Scalar(0, 0, 255);
  cv::Scalar colorMap[] = {red, red, blue, green, green};
  const uint32_t thickness = -1;
  int keyPointNums = 5;
  for (int i = 0; i < keypointList->posevec_size(); i++) {
    auto &keypointVec = keypointList->posevec(i);
    auto &object = mxpiObjectList->objectvec(i);
    float width = object.x1() - object.x0();
    float height = object.y1() - object.y0();
    for (int j = 0; j < keyPointNums; j++) {
      auto keypoint = keypointVec.keypointvec(j);
      float x = keypoint.x();
      float y = keypoint.y();
      int pixelNums = 50;
      // 每50个像素，关键点半径+1
      cv::circle(imgBgr, cv::Point(x, y),
                 std::min(width, height) / pixelNums + 1, colorMap[j],
                 thickness);
    }
  }

  MkdirRecursive("./result");
  cv::imwrite("./result/result.jpg", imgBgr);
  ret = MxBase::MemoryHelper::MxbsFree(memoryDst);
  if (ret != APP_ERR_OK) {
    LogError << "Fail to MxbsFree memory.";
    return ret;
  }
  return APP_ERR_OK;
}

// 当没有检测到人脸时，输出原始图片
static APP_ERROR CopyFile(const std::string srcFile, const std::string desFile) {
    std::ifstream is(srcFile, std::ifstream::in | std::ios::binary);
    is.seekg(0, is.end);
    int length = is.tellg();
    is.seekg(0, is.beg);
    char * buffer = new char[length];
    is.read(buffer, length);
    std::ofstream os(desFile, std::ofstream::out | std::ios::binary);
    if (!os.is_open()) {
        LogError << "Fail to open the picture.";
        return -1;
    }
    os.write(buffer, length);
    delete [] buffer;
    is.close();
    os.close();
    return APP_ERR_OK;
}

// 打印protobuf信息
static APP_ERROR PrintInfo(std::vector<MxStream::MxstProtobufOut> outPutInfo) {
  if (outPutInfo.size() == 0) {
    LogError << "outPutInfo size is 0";
    return APP_ERR_ACL_FAILURE;
  }
  LogError << "outPutInfo size is " << outPutInfo.size();
  if (outPutInfo[0].errorCode != APP_ERR_OK) {
    LogError << "GetProtobuf error. errorCode=" << outPutInfo[0].errorCode;
    return outPutInfo[0].errorCode;
  }

  for (MxStream::MxstProtobufOut info : outPutInfo) {
    LogInfo << "errorCode=" << info.errorCode;
    LogInfo << "key=" << info.messageName;
    LogInfo << "value=" << info.messagePtr.get()->DebugString();
  }

  return APP_ERR_OK;
}

int main(int argc, char *argv[]) {
  // 读取test.pipeline文件信息
  system(
      "chmod  640 ../plugins/lib/plugins/libmxpi_centerfacepostprocessor.so");
  system("chmod  640 "
         "../plugins/lib/plugins/libmxpi_centerfacekeypointpostprocessor.so");
  std::string pipelineConfigPath = "../model/CenterFace.pipeline";
  std::string pipelineConfig = ReadPipelineConfig(pipelineConfigPath);
  if (pipelineConfig == "") {
    return APP_ERR_COMM_INIT_FAIL;
  }

  std::string streamName = "center_face";
  // 新建一个流管理MxStreamManager对象并初始化
  auto mxStreamManager = std::make_shared<MxStream::MxStreamManager>();
  APP_ERROR ret = mxStreamManager->InitManager();
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Fail to init Stream manager.";
    return ret;
  }
  // 加载pipeline得到的信息，创建一个新的stream业务流
  ret = mxStreamManager->CreateMultipleStreams(pipelineConfig);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Fail to creat Stream.";
    return ret;
  }

  // 获取图片名
  std::vector<std::string> arguments = {};
  for (int i = 0; i < argc; i++) {
    arguments.emplace_back(argv[i]);
  }
  std::string fileName;
  if (arguments.size() > 1) {
    fileName = arguments[1];
  } else {
    LogError << "Please set your picture file path .";
  }

  // 将图片的信息读取到dataBuffer中
  MxStream::MxstDataInput dataBuffer;
  ret = readfile(fileName, dataBuffer);
  if (ret != APP_ERR_OK) {
    LogError << "Fail to read image file, ret = " << ret << ".";
    return ret;
  }
  // 通过SendData函数传递输入信息到指定的工作元件模块
  // streamName是pipeline文件中业务流名称；inPluginId为输入端口编号，对应输入元件的编号
  ret = mxStreamManager->SendData(streamName, 0, dataBuffer);
  if (ret != APP_ERR_OK) {
    delete dataBuffer.dataPtr;
    LogError << "Fail to send data to stream, ret = " << ret << ".";
    return ret;
  }
  delete dataBuffer.dataPtr;
  // 获得Stream上输出原件的protobuf结果
  std::vector<std::string> keyVec = {"mxpi_opencvosd0",
                                     "mxpi_objectpostprocessor0",
                                     "mxpi_objectpostprocessor1"};
  std::vector<MxStream::MxstProtobufOut> output =
      mxStreamManager->GetProtobuf(streamName, 0, keyVec);
  ret = PrintInfo(output);
  if (ret != APP_ERR_OK) {
    LogError << "Fail to print the info of output, ret = " << ret << ".";
    LogError << "Fail to detect face.";
    MkdirRecursive("./result");
    std::string desFile = "./result/result.jpg";
    CopyFile(fileName, desFile);
    return ret;
  }

  // mxpi_objectpostprocessor0模型后处理插件输出信息
  auto objectList =
      std::static_pointer_cast<MxTools::MxpiObjectList>(output[1].messagePtr);
  // mxpi_imagedecoder0图片解码插件输出信息
  auto mxpiVision =
      std::static_pointer_cast<MxTools::MxpiVisionList>(output[0].messagePtr);
  auto keypointList =
      std::static_pointer_cast<MxTools::MxpiPoseList>(output[2].messagePtr);
  // 将结果写入本地图片中
  SaveResult(mxpiVision, objectList, keypointList);

  mxStreamManager->DestroyAllStreams();

  return 0;
}
