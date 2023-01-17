/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 * Description: Complete Sample Implementation of Target Detection in C++.reate
 * Author: MindX SDK
 * Create: 2023
 * History: NA
*/

#include "MxBase/Log/Log.h"
#include "MxStream/DataType/DataHelper.h"
#include "MxStream/Stream/SequentialStream.h"

using namespace MxStream;

int main(int argc, char *argv[])
{
    std::map<std::string, std::string> props0 = {
        {"modelPath", "./data/models/yolov3/yolov3_tf_bs1_fp16.om"},
        {"postProcessConfigPath", "./data/models/yolov3/yolov3_tf_bs1_fp16.cfg"},
        {"labelPath", "./data/models/yolov3/yolov3.names"},
        {"postProcessLibPath", "libMpYOLOv3PostProcessor.so"}
    };
    std::map<std::string, std::string> props1 = {
        {"modelPath", "./data/models/resnet50/resnet50_aipp_tf.om"},
        {"postProcessConfigPath", "./data/models/resnet50/resnet50_aipp_tf.cfg"},
        {"labelPath", "./data/models/resnet50/resnet50_clsidx_to_labels.names"},
        {"postProcessLibPath", "libresnet50postprocessor.so"},
    };
    std::map<std::string, std::string> props2 = {
    {"outputDataKeys", "mxpi_modelinfer0,mxpi_modelinfer1"}
    };
    
    SequentialStream stream("stream");
    stream.SetDeviceId("0");
    
    stream.Add(PluginNode("appsrc"));
    stream.Add(PluginNode("mxpi_imagedecoder"));
    stream.Add(PluginNode("mxpi_imageresize"));
    stream.Add(PluginNode("mxpi_modelinfer", props0));
    stream.Add(PluginNode("mxpi_imagecrop"));
    stream.Add(PluginNode("mxpi_imageresize"));
    stream.Add(PluginNode("mxpi_modelinfer", props1));
    stream.Add(PluginNode("mxpi_dataserialize", props2));
    stream.Add(PluginNode("appsink"));
    
    auto ret = stream.Build();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to build stream.";
        return ret;
    }
    
    auto bufferInput = MxStream::DataHelper::ReadImage("./data/images/test.jpg");
    std::vector<MxstMetadataInput> mxstMetadataInputVec;
    ret = stream.SendData("appsrc0", mxstMetadataInputVec, bufferInput);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to send data to stream.";
        return ret;
    }
    auto output = stream.GetResult("appsink0", std::vector<std::string>());
    if (output.bufferOutput != nullptr) {
        LogInfo << "Result: "
                << std::string(reinterpret_cast<char*>(output.bufferOutput->dataPtr), output.bufferOutput->dataSize);
    }
    return 0;
}