/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description: Yolov3 infer for mxBase v2
 * Author:
 * Create: 2023
 * History: NA
 * */


#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include "opencv2/opencv.hpp"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/MxBase.h"
#include "MxBase/postprocess/include/ObjectPostProcessors/Yolov3PostProcess.h" // sdk inside YoloV3PostProcess fuc


using namespace MxBase;
using namespace std;
namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
    const uint32_t YOLOV3_RESIZE = 416;
}

struct V2Param {
    uint32_t deviceId;
    std::string labelPath;
    std::string configPath;
    std::string modelPath;
};

struct ImageInfo {
	std::string oriImagePath;
	MxBase::Image oriImage;
};

void InitV2Param(V2Param &v2Param)
{
	std::string yolov3ModelPath = "./model/yolov3_tf_bs1_fp16.om";
	std::string yolov3ConfigPath = "./model/yolov3_tf_bs1_fp16.cfg";
	std::string yolov3LabelPath = "./model/yolov3.names";
	
    v2Param.deviceId = 0;
    v2Param.labelPath = yolov3LabelPath;
    v2Param.configPath = yolov3ConfigPath;
    v2Param.modelPath = yolov3ModelPath;
};

APP_ERROR YoloV3PostProcess(ImageInfo imageInfo, std::string& yoloV3ConfigPath, std::string& yoloV3LablePath,
                        std::vector<MxBase::Tensor>& yoloV3Outputs, std::vector<MxBase::Rect>& cropConfigVec)
{
    /// This should made by user! This func only show used with sdk`s yolov3 so lib.
    std::cout << "******YoloV3PostProcess******" << std::endl;
    // make yoloV3config map
    std::map<std::string, std::string> postConfig;

    postConfig.insert(pair<std::string, std::string>("postProcessConfigPath", yoloV3ConfigPath));
    postConfig.insert(pair<std::string, std::string>("labelPath", yoloV3LablePath));

    // init postProcess
    Yolov3PostProcess yolov3PostProcess;
    APP_ERROR ret = yolov3PostProcess.Init(postConfig);
	if (ret != APP_ERR_OK)
	{
		LogError << "Initialize yolov3 post processor failed, ret=" << ret;
		return ret;
	}

    // make postProcess inputs
    vector<TensorBase> tensors;
    for (size_t i = 0; i < yoloV3Outputs.size(); i++)
    {
        MemoryData memoryData(yoloV3Outputs[i].GetData(), yoloV3Outputs[i].GetByteSize());
        TensorBase tensorBase(memoryData, true, yoloV3Outputs[i].GetShape(), TENSOR_DTYPE_INT32);
        tensors.push_back(tensorBase);
    }
    vector<vector<ObjectInfo>> objectInfos;

    auto shape = yoloV3Outputs[0].GetShape();
    MxBase::ResizedImageInfo imgInfo;
    imgInfo.widthOriginal = imageInfo.oriImage.GetOriginalSize().width;
    imgInfo.heightOriginal = imageInfo.oriImage.GetOriginalSize().height;
    imgInfo.widthResize = YOLOV3_RESIZE;
    imgInfo.heightResize = YOLOV3_RESIZE;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);
    
    // do postProcess
    ret = yolov3PostProcess.Process(tensors, objectInfos, imageInfoVec);
	if (ret != APP_ERR_OK)
	{
		LogError << "yolov3 post processor execute failed, ret=" << ret;
		return ret;
	}
	
	// get origin image
	cv::Mat imgBgr = cv::imread(imageInfo.oriImagePath);

    // print result
    std::cout << "Size of objectInfos is " << objectInfos.size() << std::endl;
    for (size_t i = 0; i < objectInfos.size(); i++)
    {
        std::cout << "objectInfo-" << i << " ,Size:"<< objectInfos[i].size() << std::endl;
        for (size_t j = 0; j < objectInfos[i].size(); j++)
        {
            std::cout << std::endl << "*****objectInfo-" << i << ":" << j << std::endl;
            std::cout << "x0 is " << objectInfos[i][j].x0 << std::endl;
            std::cout << "y0 is " << objectInfos[i][j].y0 << std::endl;
            std::cout << "x1 is " << objectInfos[i][j].x1 << std::endl;
            std::cout << "y1 is " << objectInfos[i][j].y1 << std::endl;
            std::cout << "confidence is " << objectInfos[i][j].confidence << std::endl;
            std::cout << "classId is " << objectInfos[i][j].classId << std::endl;
            std::cout << "className is " << objectInfos[i][j].className << std::endl;

			uint32_t y0 = objectInfos[i][j].y0;
			uint32_t x0 = objectInfos[i][j].x0;
			uint32_t y1 = objectInfos[i][j].y1;
			uint32_t x1 = objectInfos[i][j].x1;
			
			cv::putText(imgBgr, objectInfos[i][j].className, cv::Point(x0 + 10, y0 + 10), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255,0), 4, 8);
			cv::rectangle(imgBgr, cv::Rect(x0, y0, x1 - x0, y1 - y0), cv::Scalar(0, 255, 0), 4);
        }
    }
    
    // output result
	cv::imwrite("result.jpg", imgBgr);
	
	ret = yolov3PostProcess.DeInit();
	if (ret != APP_ERR_OK)
	{
		LogError << "yolov3 post processor deinit failed, ret=" << ret;
		return ret;
	}
    
    std::cout << "******YoloV3PostProcess end******" << std::endl;
	return ret;
};

int main(int argc, char* argv[])
{
    if (argc <= 1) 
    {
        LogWarn << "Please input image path, such as './cppv2_sample test.jpg'.";
        return APP_ERR_OK;
    }
	std::string imgPath = argv[1]; 
    APP_ERROR ret;
    
    // *****1初始化模型推理
    V2Param v2Param;
    InitV2Param(v2Param);
	int32_t deviceId = v2Param.deviceId;
	std::string modelPath = v2Param.modelPath;

    // global init
    ret = MxInit();
    if (ret != APP_ERR_OK) 
    {
        LogError << "MxInit failed, ret=" << ret << ".";
    }

    // imageProcess init
    MxBase::ImageProcessor imageProcessor(deviceId); 

    // model init
    MxBase::Model yoloV3(modelPath, deviceId);

    // *****2读取图片
    MxBase::Image decodedImage;
    ret = imageProcessor.Decode(imgPath, decodedImage, ImageFormat::YUV_SP_420);
    if (ret != APP_ERR_OK) 
    {
        LogError << "Decode failed, ret=" << ret;
        return ret;
    }

    // *****3缩放图片
    MxBase::Image resizeImage;
    // set size param
    Size resizeConfig(YOLOV3_RESIZE, YOLOV3_RESIZE);

    ret = imageProcessor.Resize(decodedImage, resizeConfig, resizeImage, Interpolation::HUAWEI_HIGH_ORDER_FILTER);
    if (ret != APP_ERR_OK) 
    {
        LogError << "Resize failed, ret=" << ret;
        return ret;
    }

    // save resize image
    std::string path = "./resized_yolov3_416.jpg";
    ret = imageProcessor.Encode(resizeImage, path);
    if (ret != APP_ERR_OK) 
    {
        LogError << "Encode failed, ret=" << ret;
        return ret;
    }

    // *****4模型推理
    Tensor tensorImg = resizeImage.ConvertToTensor();
    ret = tensorImg.ToDevice(deviceId);
    if (ret != APP_ERR_OK) 
    {
        LogError << "ToDevice failed, ret=" << ret;
        return ret;
    }

    // make infer input
    std::vector<Tensor> yoloV3Inputs = {tensorImg};
    // do infer
    std::vector<Tensor> yoloV3Outputs = yoloV3.Infer(yoloV3Inputs);
    std::cout << "yoloV3Outputs len=" << yoloV3Outputs.size() << std::endl;

    // !move result to host!
    for (auto output : yoloV3Outputs)
    {
        output.ToHost();
    }

    // *****5模型后处理
    std::vector<MxBase::Rect> cropConfigVec;
	ImageInfo imageInfo;
	imageInfo.oriImagePath = argv[1];
	imageInfo.oriImage = decodedImage;
	
    ret = YoloV3PostProcess(imageInfo, v2Param.configPath, v2Param.labelPath, yoloV3Outputs, cropConfigVec);
	if (ret != APP_ERR_OK)
	{
		LogError << "YoloV3PostProcess execute failed, ret=" << ret;
		return ret;
	}

    return APP_ERR_OK;
};