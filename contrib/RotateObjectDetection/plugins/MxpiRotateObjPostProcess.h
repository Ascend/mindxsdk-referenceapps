/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SDKMEMORY_MXPIROTATEOBJPOSTPROCESS_H 
#define SDKMEMORY_MXPIROTATEOBJPOSTPROCESS_H
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "opencv2/opencv.hpp"
#include "mxpiRotateobjProto.pb.h"

namespace {

    const float CONFIDENCE_THRESH = 0.25;
    const float IOU_THRESH = 0.4;
    const int ANCHOR_DIM = 3;
    const int YOLO_OUTPUT_SIZE = 3;
    const int WIDTHINDEX = 2;
    const int HEIGHTINDEX = 3;
    const int CLASS_NUM = 16;
    const int ANGLE_NUM = 180;
    const int BOX_DIM = 4;
    const int NET_WIDTH = 1024;
    const int NET_HEIGHT = 1024;
    const int BIASES_NUM = 18;
    const int OFFSETY = 1;
    const int SCALE = 32;
    const int BIASESDIM = 2;
    const int OFFSETWIDTH = 2;
    const int OFFSETHEIGHT = 3;
    const int OFFSETBIASES = 1;
    const int OFFSETOBJECTNESS = 1;
    const int MAX_WH = 4096;
    auto uint8Deleter = [](uint8_t* p) {};

    struct RotatedObjectInfo {
    public:
        float x_c;
        float y_c;
        float width;
        float height;
        float angle;
        float confidence;
        int classID;
        std::string className;
        cv::Mat poly;
    };

    struct OutputLayer {
        size_t width;
        size_t height;
        float anchors[6];
    };

    struct NetInfo {
        int anchorDim;
        int classNum;
        int angleNum;
        int bboxDim;
        int netWidth;
        int netHeight;
    };

}

/**
* @brief Definition of MxpiRotateObjPostProcess class.
*/
namespace MxPlugins {

class MxpiRotateObjPostProcess : public MxTools::MxPluginBase {
public:
    /**
     * @brief Initialize configure parameter.
     * @param configParamMap
     * @return APP_ERROR
     */
    APP_ERROR Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) override;

    /**
     * @brief DeInitialize configure parameter.
     * @return APP_ERROR
     */
    APP_ERROR DeInit() override;

    /**
     * @brief Process the data of MxpiBuffer.
     * @param mxpiBuffer
     * @return APP_ERROR
     */   
    APP_ERROR Process(std::vector<MxTools::MxpiBuffer*>& mxpiBuffer) override;

    /**
     * @brief Definition the parameter of configure properties.
     * @return std::vector<std::shared_ptr<void>>
     */
    static std::vector<std::shared_ptr<void>> DefineProperties();

protected:
    /**
     * @brief Get the size of anchor boxes.
     * @param strBiases - A string that holds 18 anchor box dimensions 
     * @return APP_ERROR
     */
    APP_ERROR GetBiases(std::string& strBiases);

    /**
     * @brief Compare the confidences between 2 classes and get the larger on.
     * @param classID - ClassID corresponding to the maximum class probability, initial value is -1
     * @param maxClassProb - Max class probability, initial value is 0
     * @param classProb -  Class probability that is calculated from the output of the model
     * @param classNum - ClassID corresponding to the param classProb
     */
    void CompareClassProb(int& classID, float& maxClassProb, float classProb, int classNum);

    /**
     * @brief Compare the confidences between 2 angles and get the larger on.
     * @param angleID - AngleID corresponding to the Max angle probability, and initial value is -1
     * @param maxAngleProb - Max angle probability, and initial value is 0
     * @param angleProb - Angle probability that is calculated from the output of the model
     * @param angleNum - AngleID corresponds to the param classProb
     */
    void CompareAngleProb(int& angleID, float& maxAngleProb, float angleProb, int angleNum);

     /**
     * @brief Trans longside format(x_c, y_c, longside, shortside, ¦È) to minAreaRect(x_c, y_c, width, height, ¦È).
     * @param rObjInfo - A struct that is used to save information of one rotated box
     * @param longside - The longside value of one rotated box
     * @param shortside - The shortside value of one rotated box
     * @param angle - The angle value of one rotated box
     */
    void longsideformat2cvminAreaRect(RotatedObjectInfo& rObjInfo, float longside, 
                                    float shortside, float angle);
   
     /**
     * @brief Get the output rotated object detection boxes.
     * @param tensors - Input tensors that are decoded from tensorPackageList
     * @param robjInfos - Vector that holds the information of rotated boxes
     * @param results - Vector that holds the information of rotated boxes after RNMS
     */
    void ObjectDetectionOutput(const std::vector <MxBase::TensorBase>& tensors,
                            std::vector<RotatedObjectInfo>& rObjInfos,
                            std::vector<RotatedObjectInfo>& results);
    
    /**
     * @brief Select the information of rotated boxes from output of yolo layer.
     * @param netout - Net output of the yolo layer
     * @param info - A struct that holds the information of netout layer,  
                   - including dimensions of anchor, number of classes, number of angles,
                   - input height of net and input width of net
     * @param rObjInfos - Vector that holds the information of rotated boxes
     * @param stride - Number of grids, the value equal height * weight of output layer
     * @param layer - A struct that holds the information of output layer,
                    - including output height, output width and sizes of anchor box
     */
    void SelectRotateObjInfo(std::shared_ptr<void> netout, NetInfo info, 
                            std::vector<RotatedObjectInfo>& rObjInfos,
                            int stride, OutputLayer layer);
    
     /**
     * @brief Generate bounding boxes from feature layer data.
     * @param featLayerData - Vector that holds the data pointer of every feature layer
     * @param rObjInfos - Vector that holds the information of rotated boxes
     * @param featLayerShapes - Vector that holds the shapes of every feature layer
     */
    void GenerateBbox(std::vector <std::shared_ptr<void>> featLayerData,
                    std::vector <RotatedObjectInfo>& rObjInfos,
                    const std::vector <std::vector<size_t>>& featLayerShapes);

private:
    APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer& buffer, const std::string pluginName,
                            const MxTools::MxpiErrorInfo mxpiErrorInfo);
    std::string parentName_;
    std::string descriptionMessage_;
    std::string imageResizeName_;
    std::ostringstream ErrorInfo_;
};
}
#endif //SDKMEMORY_MXPIROTATEOBJPOSTPROCESS_H
