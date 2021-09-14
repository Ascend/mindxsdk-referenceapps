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
#include <string>
#include "opencv2/opencv.hpp"
#include "MxBase/Log/Log.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "MxBase/CV/Segmentation/DrawPixels.h"
#include "MxBase/PostProcessBases/SemanticSegPostProcessBase.h"

#define INPUT_MODEL_HEIGHT 512
#define INPUT_MODEL_WIDTH 512
#define OUTPUT_MODEL_WIDTH 512
// Read the information in the file
static APP_ERROR  readfile(const std::string& filePath, MxStream::MxstDataInput& dataBuffer)
{
    char c[PATH_MAX + 1] = {0x00};
    size_t count = filePath.copy(c, PATH_MAX +1);
    if(count != filePath.length()){
        LogError << "Failed to copy file path(" << c << ").";
        return APP_ERR_COMM_FAILURE;
    }
    // Gets the absolute path to the file
    char path[PATH_MAX + 1] = { 0x00 };
    if((strlen(c) > PATH_MAX) || (realpath(c, path) == nullptr)){
        LogError << "Failed to get image, the image path is (" << filePath << ").";
        return APP_ERR_COMM_NO_EXIST;
    }
    // Open the file
    FILE *fp = fopen(path, "rb");
    if(fp == nullptr){
        LogError << "Failed to open file (" << path << ").";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // Gets the length of the file content
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // If the contents of the file are not empty, write the contents of the file to dataBuffer
    if(fileSize > 0){
        dataBuffer.dataSize = fileSize;
        dataBuffer.dataPtr = new (std::nothrow) uint32_t[fileSize];//Memory is allocated based on file length
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

// Read the pipeline information
static std::string readpipelineconfig(const std::string &pipelineConfigPath)
{
    // Open the file in binary mode
    std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
    if(!file){
        LogError << pipelineConfigPath << " file is not exists";
        return "";
    }

    // Get the file size
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();

    file.seekg(0);
    std::unique_ptr<char[]> data(new char[fileSize]);

    // Store the information read in the file in data
    file.read(data.get(), fileSize);
    file.close();
    // Stores the contents of the read file in a string and returns it out
    std::string pipelineConfig(data.get(), fileSize);
    return pipelineConfig;
}

//Gets the amount of tension
void gettensors(const MxTools::MxpiTensorPackageList tensorPackageList,std::vector<MxBase::TensorBase> &tensors) {
    for (int i = 0; i < tensorPackageList.tensorpackagevec_size(); ++i) {
        for (int j = 0; j < tensorPackageList.tensorpackagevec(i).tensorvec_size(); j++) {
            MxBase::MemoryData memoryData = {};
            memoryData.deviceId = tensorPackageList.tensorpackagevec(i).tensorvec(j).deviceid();
            memoryData.type = (MxBase::MemoryData::MemoryType)tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).memtype();
            memoryData.size = (uint32_t) tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordatasize();
            memoryData.ptrData = (void *) tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordataptr();
            std::vector<uint32_t> outputShape = {};
            for (int k = 0; k < tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensorshape_size(); ++k) {
                outputShape.push_back((uint32_t) tensorPackageList.
                        tensorpackagevec(i).tensorvec(j).tensorshape(k));
            }
            MxBase::TensorBase tmpTensor(memoryData, true, outputShape,
                                         (MxBase::TensorDataType)tensorPackageList.
                                                 tensorpackagevec(i).tensorvec(j).tensordatatype());
            tensors.push_back(tmpTensor);
        }
    }
}

void semanticsegoutput(const std::vector<MxBase::TensorBase>& tensors,
                       const std::vector<MxBase::ResizedImageInfo>& resizedImageInfos,
                       std::vector<MxBase::SemanticSegInfo> &semanticSegInfos)
{
    auto tensor = tensors[0];
    auto shape = tensor.GetShape();//4
    uint32_t batchSize = shape[0];//1
    int classNum_ = 1;//float32
    // NCHW type is not supported yet.NHWC
    for (uint32_t i = 0; i < batchSize; i++) {
        uint32_t inputModelHeight = resizedImageInfos[i].heightResize;
        uint32_t inputModelWidth = resizedImageInfos[i].widthResize;
        uint32_t outputModelWidth = OUTPUT_MODEL_WIDTH;
        MxBase::SemanticSegInfo semanticSegInfo;
        //The first address of the picture data
        auto tensorPtr = (float*)tensor.GetBuffer() + i * tensor.GetByteSize() / batchSize;
        std::vector<std::vector<int>> results(inputModelHeight, std::vector<int>(inputModelWidth));
        int count = 0;
        for (uint32_t y = 0; y < inputModelHeight; y++) {
            for (uint32_t x = 0; x < inputModelWidth; x++) {
                float* begin = tensorPtr +  y * outputModelWidth * classNum_ + x * classNum_;
                results[y][x] = (*begin)*2;
                count++;
            }
        }
        semanticSegInfo.pixels = results;//Information about a picture
        semanticSegInfos.push_back(semanticSegInfo);
    }
}

//Mask diagram generation
APP_ERROR draw(const std::vector<MxBase::TensorBase>& tensors,
               std::vector<MxBase::SemanticSegInfo>& semanticSegInfos,
               const std::vector<MxBase::ResizedImageInfo>& resizedImageInfos,
               std::string & inputPicname)
{
    int dumpImage_ = 1;
    auto inputs = tensors;
    semanticsegoutput(inputs, resizedImageInfos, semanticSegInfos);
    if (dumpImage_) {
        for (uint32_t i = 0; i < semanticSegInfos.size(); i++) {
            std::ostringstream outputPath;
            outputPath << "./data/mask_" << inputPicname;
            MxBase::DrawPixelsRGB(semanticSegInfos[i].pixels, outputPath.str());
        }
    }
    return APP_ERR_OK;
}

//Mask chart zoom
void zoom(std::string filename,int height,int width){
    cv::Mat src = cv::imread("./"+filename,cv::IMREAD_UNCHANGED);
    cv::Mat dst;
    resize(src, dst, cv::Size(width, height));//缩放图像
    cv::imwrite(filename,dst);
}

//Picture fusion
void  image_fusion(std::string filename,std::string maskname,std::string &inputPicname){
    cv::Mat img1 = cv::imread(filename);
    cv::Mat img2 = cv::imread(maskname);
    cv::Mat dst;
    //1 and 0.5 are the transparency of array
    //0 means offset added to weighted sum
    cv::addWeighted(img1,1,img2,0.5,0,dst);
    cv::imwrite("./result/result_"+inputPicname,dst);
}

int main(int argc, char* argv[])
{
    //Enter the image name, path
    std::string inputPicname = "test.jpg";
    std::string inputPicPath = "./data/"+inputPicname;
    unsigned long idx = inputPicname.find(".jpg");

    if(idx == std::string::npos ){
        LogError << "The input is incorrect\n";
        return 0;
    }

    // Read the test.pipeline file information
    std::string pipelineConfigPath = "./test.pipeline";
    std::string pipelineConfig = readpipelineconfig(pipelineConfigPath);
    if(pipelineConfig == ""){
        return APP_ERR_COMM_INIT_FAIL;
    }
    std::string streamName = "detection";
    // Create a new stream management MxStreamManager object and initialize it
    auto mxStreamManager = std::make_shared<MxStream::MxStreamManager>();
    APP_ERROR ret = mxStreamManager->InitManager();//Initialize the flow management tool
    if(ret != APP_ERR_OK){
        LogError << GetError(ret) << "Fail to init Stream manager.";
        return ret;
    }
    // Load the information that pipeline gets to create a new stream business flow
    ret = mxStreamManager->CreateMultipleStreams(pipelineConfig);//The incoming profile
    if(ret != APP_ERR_OK){
        LogError << GetError(ret) << "Fail to creat Stream.";
        return ret;
    }

    // Read the information of the picture into dataBuffer
    MxStream::MxstDataInput dataBuffer;
    ret = readfile(inputPicPath, dataBuffer);
    if(ret != APP_ERR_OK){
        LogError << "Fail to read image file, ret = " << ret << ".";
        return ret;
    }
    // The input information is passed through the SendData function to the specified working element module
    // StreamName is the name of the business flow in the pipeline file
    ret = mxStreamManager->SendData(streamName, 0, dataBuffer);
    if(ret != APP_ERR_OK){
        delete dataBuffer.dataPtr;
        LogError << "Fail to send data to stream, ret = " << ret << ".";
        return ret;
    }
    delete dataBuffer.dataPtr;

    std::vector<std::string> keyVec = {"mxpi_tensorinfer0", "mxpi_imagedecoder0"};
    std::vector<MxStream::MxstProtobufOut> output = mxStreamManager->GetProtobuf(streamName, 0, keyVec);

    // Mxpi_tensorinfer0 model post-processing plug-in output information
    auto objectList = std::static_pointer_cast<MxTools::MxpiTensorPackageList>(output[0].messagePtr);
    // mxpi_imagedecoder0 the image decoding plug-in output information
    auto mxpiVision = std::static_pointer_cast<MxTools::MxpiVisionList>(output[1].messagePtr);
    auto tensorPackageList = google::protobuf::DynamicCastToGenerated<MxTools::MxpiTensorPackageList>
            (output[0].messagePtr.get());
    int Pre_Height = mxpiVision.get()->visionvec(0).visioninfo().height();
    int Pre_Width = mxpiVision.get()->visionvec(0).visioninfo().width();
    MxTools::MxpiTensorPackage tensorPackage = tensorPackageList->tensorpackagevec(0);
    MxTools::MxpiTensor tensor = tensorPackage.tensorvec(0);
    std::vector<MxBase::TensorBase> tensors;

    gettensors(*tensorPackageList,tensors);
    std::vector<MxBase::ResizedImageInfo> ResizedImageInfos;
    std::vector<MxBase::SemanticSegInfo> semanticSegInfos;
    MxBase::ResizedImageInfo resizedImageInfo;//输入图片数据信息
    resizedImageInfo.heightResize = INPUT_MODEL_HEIGHT;
    resizedImageInfo.widthResize = INPUT_MODEL_WIDTH;
    ResizedImageInfos.push_back(resizedImageInfo);
    draw(tensors,semanticSegInfos,ResizedImageInfos,inputPicname);
    zoom("./data/mask_"+inputPicname,Pre_Height,Pre_Width);
    image_fusion(inputPicPath,"./data/mask_"+inputPicname,inputPicname);

    mxStreamManager->DestroyAllStreams();
    return 0;
}








