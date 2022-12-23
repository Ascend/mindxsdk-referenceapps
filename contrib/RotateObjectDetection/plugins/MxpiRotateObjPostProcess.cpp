/*
* Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include "MxpiRotateObjPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "MxBase/Maths/FastMath.h"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
using namespace cv;

namespace {

    auto uint8Deleter = [](uint8_t* p) {};
    std::string strBiases = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326";
    const std::string className[16] = {"plane", "baseball-diamond", "bridge", "ground-track-field", 
                                "small-vehicle", "large-vehicle", "ship", "tennis-court",
                                "basketball-court", "storage-tank",  "soccer-ball-field", "roundabout", 
                                "harbor", "swimming-pool", "helicopter", "container-crane"};
    std::vector<float> biases_ = {};
    const double ZERO = 1E-8;
    const int MAXN = 51;
    const int MAX_DET = 1000; 
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
    const int ZERODEGREEANGLE = 0;
    const int RIGHTANGLE = 90;
    const int FLATANGLE = 180;
    const float MAPPINGANGLE = 179.9;

}

namespace Rnms{

/**
 * @brief Determine the signal of a double-precision data.
 * @param s - A double-precison data
 * @return 0, if the data is in the range of [-1E-8, 1E-8] 
 *         1, if the date is greater than 1E-8
 *        -1, if the data is smaller than -1E-8 
 */
static int Sign(double s) {
    if (s > ZERO){
        return 1;
    }
    else if (s < -ZERO){
        return -1;
    }
    else {
        return 0;
    }
}

/**
 * @brief Calculate the cross product of two vectors oa and ob.
 * Since the cross product computation formula is |a||b|sin(¦È), 
 * the cross product represents the area of signed triangle oab.
 * @param o - cv::Point2d 
 * @param a - cv::Point2d
 * @param b - cv::Point2d
 * @return Cross product of vector oa and vector ob
 */
static double Cross(cv::Point2d o, cv::Point2d a, cv::Point2d b) {
    double crossValue = (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
    return crossValue;
}

/**
 * @brief Calculate the area of one polygon.
 * @param poly - An array of type cv::Point2d that holds coordinates of polygon 
 * @param n - The length of array poly, i.e. the number of sides of a polygon
 * @return Area of a polygon
 */
static double PolygonArea(cv::Point2d* poly, int n) {
    poly[n] = poly[0];
    double area = 0;
    for (int i = 0; i < n; i++) {
        area += (poly[i].x * poly[i+1].y - poly[i].y * poly[i+1].x) / 2.0;
    }
    return area;

}

/**
 * @brief Calculate the coordinates of the intersection of two lines ab and cd, and store the value in point p.
 * @param a - cv::Point2d
 * @param b - cv::Point2d
 * @param c - cv::Point2d
 * @param d - cv::Point2d
 * @param p - cv::Point2d
 * @return 0, if point a, b, c and d are on a line
 *         1, if line ab is parallel to cd
 *         2, if line ab intersects cd, and the intersection coordinates are stored in point p 
 */
static int CalcLineIntersect(cv::Point2d a, cv::Point2d b, cv::Point2d c, cv::Point2d d, cv::Point2d& p) {
    double s1 = 0;
    double s2 = 0;
    s1 = Cross(a, b, c);
    s2 = Cross(a, b, d);
    if (Sign(s1) == 0 && Sign(s2) == 0){
        return 0;
    }        
    if (Sign(s2 - s1) == 0){
        return 1;
    }        
    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    return 2;
}

/**
 * @brief Line ab cuts the polygon p, and the points at the upper left of the line are saved.
 * @description:If the cross value of vector ab, a poly[i] is positive, it shows that the point poly[i] is in the 
                upper of the line, so this point should be saved into array pp.
                If the cross value of vector ab, a poly[i] is not equal to the cross value of vector ab, a poly[i+1],
                it shows that line ab intersects line poly[i]poly[i+1], so we should calculate the intersection point 
                and save it into array pp. 
 * @param poly - An array of type cv::Point2d that holds coordinates of polygon
 * @param n - The length of array poly, i.e. the number of sides of a polygon
 * @param a - cv::Point2d
 * @param pp - An array of type cv::Point2d that save the cut points temporarily
 */
static void PolygonCut(cv::Point2d* poly, int& n, cv::Point2d a, cv::Point2d b, cv::Point2d* pp) {
    int m = 0;
    poly[n] = poly[0];
    for (int i = 0; i < n; i++) {
        if (Sign(Cross(a, b, poly[i])) > 0){
            pp[m++] = poly[i];
        }             
        if (Sign(Cross(a, b, poly[i])) != Sign(Cross(a, b, poly[i + 1]))){
            CalcLineIntersect(a, b, poly[i], poly[i+1], pp[m++]);
        }         
    }
    n = 0;
    for (int i = 0; i < m; i++){
        if (!i || !(pp[i] == pp[i-1])){
            poly[n++] = pp[i];
        }        
    }
    while (n > 1 && poly[n-1] == poly[0]){
        n--;
    }        
}

/**
 * @brief Calculate the area intersection of triange oab and triangle ocd.
 * @param a - cv::Point2d
 * @param b - cv::Point2d
 * @param c - cv::Point2d
 * @param d - cv::Point2d
 * @return The value of intersection area of two triangles
 */
static double IntersectArea(cv::Point2d a, cv::Point2d b, cv::Point2d c, cv::Point2d d) {
    cv::Point2d o(0, 0);
    int s1 = Sign(Cross(o, a, b));
    int s2 = Sign(Cross(o, c, d));
    // The signed area of triangle oab or of triangle ocd is equal to 0, return 0.
    if ((s1 == 0) || (s2 == 0)){
        return 0.0;
    }
    // The signed area of triangle oab is negative, swap a, b.
    if (s1 == -1){
        swap(a, b);
    }   
    // The signed area of triangle ocd is negative, swap c, d.
    if (s2 == -1){
        swap(c, d);
    }    
    cv::Point2d p[10] = {o, a, b};
    int n = 3;
    cv::Point2d pp[MAXN];
    // Cut the triangle oab with line oc, cd, do respectively.
    PolygonCut(p, n, o, c, pp);
    PolygonCut(p, n, c, d, pp);
    PolygonCut(p, n, d, o, pp);
    // Calculate the overlap area.
    double interAreaValue = fabs(PolygonArea(p, n));
    if (s1 * s2 == -1){
        interAreaValue = -interAreaValue;
    }        
    return interAreaValue;
}

/**
 * @brief Calculate the area intersection of polygon1 and polygon2.
 * @param poly1 - An array of type cv::Point2d that holds coordinates of polygon1
 * @param n1 - The length of array poly1, i.e. the number of sides of a polygon1
 * @param poly2 - An array of type cv::Point2d that holds coordinates of polygon2
 * @param n2 - The length of array poly2, i.e. the number of sides of a polygon2
 * @return The value of intersection area of two polygons
 */
static double IntersectArea(cv::Point2d* poly1, int n1, cv::Point2d* poly2, int n2) {
    if (PolygonArea(poly1, n1) < 0){
        reverse(poly1, poly1 + n1);
    }     
    if (PolygonArea(poly2, n2) < 0){
        reverse(poly2, poly2 + n2);
    }       
    poly1[n1] = poly1[0];
    poly2[n2] = poly2[0];
    double interAreaValue = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            interAreaValue += IntersectArea(poly1[i], poly1[i+1], poly2[j], poly2[j+1]);
        }
    }
    return interAreaValue;
}

/**
 * @brief Convert the matrix form to array form of type cv::Point2d.
 * @param m - A matrix that holds coordinates of polygon
 * @param p - An array of type cv::Point2d that holds coordinates of polygon 
 */
static void Mat2Point(cv::Mat M, cv::Point2d p[]) {
    cv::Mat tmp;
    int index = 0;
    M.convertTo(tmp, CV_64FC1);
    for (int row = 0; row < tmp.rows; row++) {
        for (int col = 0; col < tmp.cols; col++) {
            if (col == 0) {
                p[index].x = tmp.at<double>(row, col);
            }
            else {
                p[index].y = tmp.at<double>(row, col);
            }
        }
        index++;
    }
}

/**
 * @brief Calculate the iou value of coordinates matrix1 and matrix2.
 * @param M1 - A matrix that holds coordinates of polygon1
 * @param M2 - A matrix that holds coordinates of polygon2 
 * @return IOU value of polygon1 and polygon2
 */
static double CalcPolyIou(cv::Mat M1, cv::Mat M2) {
    cv::Point2d poly1[MAXN], poly2[MAXN];
    int n1 = 4;
    int n2 = 4;
    Mat2Point(M1, poly1);
    Mat2Point(M2, poly2);
    double interArea = IntersectArea(poly1, n1, poly2, n2);
    double unionArea = fabs(PolygonArea(poly1, n1)) + fabs(PolygonArea(poly2, n2)) - interArea;
    double iou = interArea / unionArea;
    return iou;
}

/**
 * @brief Comparation between two RotatedObjectInfo elements.
 * @param rObjInfo1 - RotatedObjectInfo rObjInfo1
 * @param rObjInfo2 - RotatedObjectInfo rObjInfo2
 * @return True if the confidence of rObjInfo1 is greater than that of rObjInfo2
 */
static bool SortConfidence(RotatedObjectInfo rObjInfo1, RotatedObjectInfo rObjInfo2) {
    if (rObjInfo1.confidence > rObjInfo2.confidence){
        return true;
    }
    else{
        return false;
    }

}

/**
 * @brief Rotated non-maximum suppression, remove overlapping bboxes.
 * When the same object is detected, keep the box with maximum confidence.
 * @param rObjInfos - Vector that holds the information of rotated boxes for RNMS
 * @param threshold - RNMS threshold
 * @return The final reserved bboxes after RNMS
 */
static std::vector<RotatedObjectInfo> Rnms(std::vector<RotatedObjectInfo>& rObjInfos, float threshold) {
    // Init a vector, and reserve a block of memory with the size MAX_DET
    std::vector<RotatedObjectInfo> results;
    results.reserve(MAX_DET);
    // Sort the vector rObjInfos in descending order of confidence
    std::sort(rObjInfos.begin(), rObjInfos.end(), SortConfidence);
    // Init a multimap, and place box and corresponding ordinal number
    std::multimap<int, RotatedObjectInfo> boxes;
    int num_boxes = rObjInfos.size();
    for (int i = 0; i < num_boxes; i++){
        boxes.insert(make_pair(i, rObjInfos[i]));
    }
    while(boxes.size() > 1){
        // Place the box with maximum confidence into vector results
        results.push_back(boxes.begin()->second);
        // it_second point to box with the second largest confidence
        std::multimap<int, RotatedObjectInfo>::iterator it_second = ++boxes.begin();
        // firstIdx represent the ordinal number corresponding to largest confidence box
        int firstIdx = boxes.begin()->first;
        // Iterate through boxes from the second iterator, and calculates the IOU value with the first box
        for (std::multimap<int, RotatedObjectInfo>::iterator it = it_second; it != boxes.end();){           
            int itIdx = it->first;
            double iou_value = CalcPolyIou(rObjInfos[firstIdx].poly, rObjInfos[itIdx].poly);
            // If IOU value is greater than threshold, erase this box and iterator point to the next box
            if(iou_value > threshold){
                it = boxes.erase(it);
            }else{
                it++;
            }
        }
        // erase the first box
        boxes.erase(boxes.begin());
    }
    if(boxes.size() == 1){
        results.push_back(boxes.begin()->second);
    }       
    return results;
}

}

namespace MxPlugins{

/**
 * @brief decode MxpiTensorPackageList
 * @param tensorPackageList - Source tensorPackageList
 * @param tensors - Target TensorBase data
 */
static void GetTensors(const MxTools::MxpiTensorPackageList tensorPackageList,
                       std::vector<MxBase::TensorBase> &tensors) {
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

/**
 * @brief Decode MxpiVisionList, get image resize Information
 * @param visionList - Source MxpiVisionList
 * @param visionInfos - Target vector data
 */
static void GetImageResizeInfo(const MxTools::MxpiVisionList visionList, std::vector<float> &visionInfos){
    MxpiVision vision = visionList.visionvec(0);
    MxpiVisionInfo visionInfo = vision.visioninfo();
    visionInfos.push_back(visionInfo.height());
    visionInfos.push_back(visionInfo.width());
    visionInfos.push_back(visionInfo.heightaligned());
    visionInfos.push_back(visionInfo.widthaligned());
    visionInfos.push_back(visionInfo.keepaspectratioscaling());
}

/**
 * @brief Map the coordinates of the image after resize to the original size 
 * @param results - A vector that holds rotated boxes after RNMS
 * @param visionInfos - A vector that holds the image resize information
 */
static void CoordinateMapping(std::vector<RotatedObjectInfo>& results, std::vector<float> &visionInfos){
    float keepAspectRatioScaling = visionInfos[4];
    for(int i = 0; i < results.size(); i++){
        cv::RotatedRect box(cv::Point(results[i].x_c, results[i].y_c), 
                            cv::Size(results[i].width, results[i].height), results[i].angle);
        cv::Mat poly;
        cv::Mat tmp;
        boxPoints(box, poly);
        poly.convertTo(tmp, CV_32FC1);
        for(int i = 0; i < poly.rows; i++)
        {
            for(int j = 0; j < poly.cols; j++)
            {
                tmp.at<float>(i, j) /= keepAspectRatioScaling;

            }
        }
        cv::RotatedRect rect = cv::minAreaRect(tmp);
        results[i].x_c = rect.center.x;
        results[i].y_c = rect.center.y;
        results[i].width = rect.size.width;
        results[i].height = rect.size.height;
        results[i].angle = rect.angle;
    }
}

/**
* @brief Initialize configure parameter.
* @param configParamMap
* @return APP_ERROR
*/
APP_ERROR MxpiRotateObjPostProcess::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "MxpiRotateObjPostProcess::Init start.";
    APP_ERROR ret = APP_ERR_OK;

    // Get the property values by key
    parentName_ = dataSource_;
    std::shared_ptr<string> imageResizePropSptr = std::static_pointer_cast<string>(configParamMap["imageSource"]);
    imageResizeName_ = *imageResizePropSptr.get();

    // Initialize the anchor size
    ret = GetBiases(strBiases);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to get biases.";
        return ret;
    }

    return APP_ERR_OK;
}

/**
* @brief DeInitialize configure parameter.
* @return APP_ERROR
*/
APP_ERROR MxpiRotateObjPostProcess::DeInit()
{
    LogInfo << "MxpiRotateObjPostProcess::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiRotateObjPostProcess::SetMxpiErrorInfo(MxpiBuffer& buffer, 
                                                     const std::string pluginName,
                                                     const MxpiErrorInfo mxpiErrorInfo)
{
    APP_ERROR ret = APP_ERR_OK;
    // Define an object of MxpiMetadataManager
    MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(pluginName, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

/**
* @brief Compare the confidences between 2 classes and get the larger on.
* @param classID - ClassID corresponding to the maximum class probability, initial value is -1
* @param maxClassProb - Max class probability, initial value is 0
* @param classProb -  Class probability that is calculated from the output of the model
* @param classNum - ClassID corresponding to the param classProb
*/
void MxpiRotateObjPostProcess::CompareClassProb(int& classID, float& maxClassProb, 
                                                float classProb, int classNum) {
    if (classProb > maxClassProb) {
        maxClassProb = classProb;
        classID = classNum;
    }
}

/**
* @brief Compare the confidences between 2 angles and get the larger on.
* @param angleID - AngleID corresponding to the maximum angle probability, and initial value is -1
* @param maxAngleProb - Max angle probability, and initial value is 0
* @param angleProb - Angle probability that is calculated from the output of the model
* @param angleNum - AngleID corresponds to the param classProb
*/
void MxpiRotateObjPostProcess::CompareAngleProb(int& angleID, float& maxAngleProb, 
                                                float angleProb, int angleNum) {
    if (angleProb > maxAngleProb) {
        maxAngleProb = angleProb;
        angleID = angleNum;
    }
}

/**
* @brief Trans longside format(x_c, y_c, longside, shortside, ¦È) to minAreaRect(x_c, y_c, width, height, ¦È).
* @param rObjInfo - A struct that is used to save information of one rotated box
* @param longside - The longside value of one rotated box
* @param shortside - The shortside value of one rotated box
* @param angle - The angle value of one rotated box
*/
void MxpiRotateObjPostProcess::longsideformat2cvminAreaRect(RotatedObjectInfo& rObjInfo, 
                                                            float longside, float shortside, float angle) {
    if ((angle >= -FLATANGLE) && (angle < -RIGHTANGLE)) {
        rObjInfo.width = shortside;
        rObjInfo.height = longside;
        rObjInfo.angle = angle + RIGHTANGLE;
    }
    else {
        rObjInfo.width = longside;
        rObjInfo.height = shortside;
        rObjInfo.angle = angle;
    }
    if ((rObjInfo.angle < -RIGHTANGLE) || (rObjInfo.angle >= ZERODEGREEANGLE)) {
        std::cout << "The current Angle is outside the scope defined by OpenCV!" << std::endl;
    }
}

/**
* @brief Get the output rotated object detection boxes.
* @param tensors - Input tensors that are decoded from tensorPackageList
* @param robjInfos - Vector that holds the information of rotated boxes
* @param results - Vector that holds the information of rotated boxes after RNMS
*/
void MxpiRotateObjPostProcess::ObjectDetectionOutput(const std::vector <MxBase::TensorBase>& tensors,
                                                     std::vector<RotatedObjectInfo>& rObjInfos,
                                                     std::vector<RotatedObjectInfo>& results) {
    LogDebug << "RotateObjectPostProcess start to write results.";
    if (tensors.size() == 0) {
        return;
    }
    auto shape = tensors[0].GetShape();
    if (shape.size() == 0) {
        return;
    }
    uint32_t batchSize = shape[0];
    // Get data and shape of feature layer
    // featLayerData stores the initial address of each tensor
    // featLayerShapes store the shape of each feature layer
    for (uint32_t i = 0; i < batchSize; i++) {
        std::vector <std::shared_ptr<void>> featLayerData = {};
        std::vector <std::vector<size_t>> featLayerShapes = {};
        for (uint32_t j = 0; j < tensors.size(); j++) {
            auto dataPtr = (uint8_t *)tensors[j].GetBuffer();
            std::shared_ptr<void> tmpPointer;
            tmpPointer.reset(dataPtr, uint8Deleter);
            featLayerData.push_back(tmpPointer);
            shape = tensors[j].GetShape();
            std::vector <size_t> featLayerShape = {};
            for (auto s : shape) {
                featLayerShape.push_back((size_t)s);
            }
            featLayerShapes.push_back(featLayerShape);
        }
        // Generate bboxes from feature layer data.
        MxpiRotateObjPostProcess::GenerateBbox(featLayerData, rObjInfos, featLayerShapes);
        // RNMS
        results = Rnms::Rnms(rObjInfos, IOU_THRESH); 
    }
    LogDebug << "RotateObjectPostProcess write results successed.";
}

/**
* @brief Select the information of rotated boxes from output of yolo layer.
* @description: According to the structure of YOLO output layer, feature layer data format is 
                [batchsize, anchorDim, width, height, (bboxDim + 1 + classNum + angleNum)],
                such as 1*3*128*128*201, 1*3*64*64*201, 1*3*32*32*201.
                To get the data index, iterate over each anchor firstly, then iterate over each grid.
* @param netout - Net output of the yolo layer
* @param info - A struct that holds the information of netout layer,  
            - including dimensions of anchor, number of classes, number of angles,
            - input height of net and input width of net
* @param rObjInfos - Vector that holds the information of rotated boxes
* @param stride - Number of grids, the value equal height * weight of output layer
* @param layer - A struct that holds the information of output layer,
            - including output height, output width and sizes of anchor box
*/
void MxpiRotateObjPostProcess::SelectRotateObjInfo(std::shared_ptr<void> netout, NetInfo info, 
                                                   std::vector<RotatedObjectInfo>& rObjInfos,
                                                   int stride, OutputLayer layer) {
    for (int j = 0; j < info.anchorDim; ++j) {
        for (int k = 0; k < stride; ++k) {
            
            // begin index
            // info.bboxDim + 1 + info.classNum + info.angleNum = 201 
            // 201 corresponds to the parameters contained in each bbox
            int bIdx = (info.bboxDim + 1 + info.classNum + info.angleNum) * stride * j + 
                         k * (info.bboxDim + 1 + info.classNum + info.angleNum);                  
            // objectness index
            int oIdx = bIdx + info.bboxDim; 
            float objectness = fastmath::sigmoid(static_cast<float*>(netout.get())[oIdx]);

            // Get classID
            int classID = -1;
            float maxClassProb = 0;
            float classProb = 0;
            for (int c = 0; c < info.classNum; ++c) {
                classProb = fastmath::sigmoid(static_cast<float*>(netout.get())[bIdx + 
                            (info.bboxDim + OFFSETOBJECTNESS + c)]) * objectness; 
                CompareClassProb(classID, maxClassProb, classProb, c);
            }
            if (classID < 0){
                continue;
            }                
            if (maxClassProb <= CONFIDENCE_THRESH) {
                continue;
            }

            // Get angleID
            int angleID = -1;
            float maxAngleProb = 0;
            float angleProb = 0;
            for (int a = 0; a < info.angleNum; ++a) {
                angleProb = fastmath::sigmoid(static_cast<float*>(netout.get())[bIdx + 
                            (info.bboxDim + OFFSETOBJECTNESS + info.classNum + a)]); 
                CompareAngleProb(angleID, maxAngleProb, angleProb, a);
            }
            if (angleID < 0){
                continue;
            } 
                
 
            // k is the number of grids
            // layer.width 128 64 32
            // k / layer.width, get the rows
            // k % layer.width, get the columns
            int row = k / layer.width;
            int col = k % layer.width;

            // Get the actual prediction of x, y, longside, shortside
            float x = (col + (fastmath::sigmoid(static_cast<float*>(netout.get())[bIdx])) 
                * 2 - 0.5) * (info.netWidth / layer.width);
            float y = (row + (fastmath::sigmoid(static_cast<float*>(netout.get())[bIdx + OFFSETY])) 
                * 2 - 0.5) * (info.netHeight / layer.height);
            float longside = std::pow(fastmath::sigmoid(static_cast<float*>(netout.get())[bIdx + OFFSETWIDTH]) 
                * 2, 2) * layer.anchors[BIASESDIM * j];
            float shortside = std::pow(fastmath::sigmoid(static_cast<float*>(netout.get())[bIdx + OFFSETHEIGHT]) 
                * 2, 2) * layer.anchors[BIASESDIM * j + OFFSETBIASES];

            // Assign to rObjInfo center point, classID, confidence
            RotatedObjectInfo rObjInfo;
            rObjInfo.x_c = x;
            rObjInfo.y_c = y;
            rObjInfo.classID = classID;
            rObjInfo.confidence = maxClassProb;
            // Assign to rObjInfo w, h, angle
            longsideformat2cvminAreaRect(rObjInfo, longside, shortside, angleID - MAPPINGANGLE);

            // Calculate polygon coordinates, and assign to rObjInfo poly
            cv::RotatedRect box(cv::Point(rObjInfo.x_c + classID * MAX_WH, rObjInfo.y_c + classID * MAX_WH), 
                                cv::Size(rObjInfo.width, rObjInfo.height), rObjInfo.angle);
            cv::Mat rObjPoly;
            boxPoints(box, rObjPoly);
            rObjInfo.poly = rObjPoly;

            // Get the category based on classID
            rObjInfo.className = className[classID];

            rObjInfos.push_back(rObjInfo);
        }
        
    }
}

/**
* @brief Generate bounding boxes from feature layer data.
* @param featLayerData - Vector that holds the data pointer of every feature layer
* @param rObjInfos - Vector that holds the information of rotated boxes
* @param featLayerShapes - Vector that holds the shapes of every feature layer
*/
void MxpiRotateObjPostProcess::GenerateBbox(std::vector <std::shared_ptr<void>> featLayerData,
                                            std::vector <RotatedObjectInfo>& rObjInfos,
                                            const std::vector <std::vector<size_t>>& featLayerShapes) {

    NetInfo netInfo;
    netInfo.anchorDim = ANCHOR_DIM;
    netInfo.bboxDim = BOX_DIM; 
    netInfo.classNum = CLASS_NUM;
    netInfo.angleNum = ANGLE_NUM; 
    netInfo.netWidth = NET_WIDTH; 
    netInfo.netHeight = NET_HEIGHT;

    for (int i = 0; i < YOLO_OUTPUT_SIZE; ++i) { 
        int widthIndex_ = WIDTHINDEX;
        int heightIndex_ = HEIGHTINDEX;
        OutputLayer layer = { featLayerShapes[i][widthIndex_], featLayerShapes[i][heightIndex_] };
        int logOrder = log(featLayerShapes[i][widthIndex_] * SCALE / NET_WIDTH) / log(BIASESDIM); 
        int startIdx = (YOLO_OUTPUT_SIZE - 1 - logOrder) * netInfo.anchorDim * BIASESDIM;  
        int endIdx = startIdx + netInfo.anchorDim * BIASESDIM;
        int idx = 0;
        for (int j = startIdx; j < endIdx; ++j) {
            layer.anchors[idx++] = biases_[j];
        }
        int stride = layer.width * layer.height; // 13*13 26*26 52*52
        std::shared_ptr<void> netout = featLayerData[i];
        SelectRotateObjInfo(netout, netInfo, rObjInfos, stride, layer);
    }
}

 /**
* @brief Get the size of anchor boxes.
* @param strBiases - A string that holds 18 anchor box dimensions 
* @return APP_ERROR
*/
APP_ERROR MxpiRotateObjPostProcess::GetBiases(std::string& strBiases) {
    if (BIASES_NUM <= 0) {
        LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "Failed to get biasesNum (" << BIASES_NUM << ").";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    biases_.clear();
    int i = 0;
    int num = strBiases.find(",");
    while (num >= 0 && i < BIASES_NUM) {
        std::string tmp = strBiases.substr(0, num);
        num++;
        strBiases = strBiases.substr(num, strBiases.size());
        biases_.push_back(stof(tmp));
        i++;
        num = strBiases.find(",");
    }
    if (i != BIASES_NUM - 1 || strBiases.size() <= 0) {
        LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "biasesNum (" << BIASES_NUM
            << ") is not equal to total number of biases (" << strBiases << ").";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    biases_.push_back(stof(strBiases));
    return APP_ERR_OK;
}

/**
* @brief Generate MxpiRotateobjList.
* @param rObjInfos - Vector that holds the information of rotated boxes
* @param results - Vector that holds the information of rotated boxes after RNMS
* @param visionInfos - A vector that holds the image resize information
* @param tensorPackageList - Source tensorPackageList
* @param mxpiRotateobjList - Target MxpiRotateobjList that holds detection result list        
*/
void MxpiRotateObjPostProcess::GenerateMxpiRotateobjList(std::vector <RotatedObjectInfo> rObjInfos,
    std::vector <RotatedObjectInfo> results,
    std::vector<float> visionInfos,
    const MxTools::MxpiTensorPackageList tensorPackageList,
    mxpirotateobjproto::MxpiRotateobjProtoList &mxpiRotateobjList){
    
    // Decode tensorPackageList to get the input.
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(tensorPackageList, tensors);
    auto inputs = tensors;
    
    // Get the output results.
    ObjectDetectionOutput(inputs, rObjInfos, results);

    // Map the coordinates to the original size.
    CoordinateMapping(results, visionInfos);
    
    // Assign to mxpiRotateobjList
    for (int i = 0; i < results.size(); i++) {

        auto mxpiRotateobjProtoptr = mxpiRotateobjList.add_rotateobjprotovec();
        mxpirotateobjproto::MxpiMetaHeader* dstMxpiMetaHeaderList = mxpiRotateobjProtoptr->add_headervec();
        dstMxpiMetaHeaderList->set_datasource(parentName_);
        dstMxpiMetaHeaderList->set_memberid(0);    
        
        mxpiRotateobjProtoptr->set_x_c(results[i].x_c);
        mxpiRotateobjProtoptr->set_y_c(results[i].y_c);
        mxpiRotateobjProtoptr->set_width(results[i].width);
        mxpiRotateobjProtoptr->set_height(results[i].height);
        mxpiRotateobjProtoptr->set_angle(results[i].angle);
        mxpiRotateobjProtoptr->set_confidence(results[i].confidence);
        mxpiRotateobjProtoptr->set_classid(results[i].classID);
        mxpiRotateobjProtoptr->set_classname(results[i].className);  
    }
}

/**
* @brief Process the data of MxpiBuffer.
* @param mxpiBuffer
* @return APP_ERROR
*/
APP_ERROR MxpiRotateObjPostProcess::Process(std::vector<MxpiBuffer*>& mxpiBuffer){
    LogInfo << "MxpiRotateObjPostProcess::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) 
            << "MxpiRotateObjPostProcess process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiRotateObjPostProcess process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the data from buffer
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }
    // Get image resize information
    shared_ptr<void> ir_metadata = mxpiMetadataManager.GetMetadata(imageResizeName_);
    shared_ptr<MxpiVisionList> imageResizeVisionListSptr = static_pointer_cast<MxpiVisionList>(ir_metadata);
    std::vector<float> visionInfos = {};
    GetImageResizeInfo(*imageResizeVisionListSptr, visionInfos); 
    // Get MxpiRotateobjList
    shared_ptr<MxpiTensorPackageList> srcMxpiTensorPackageListSptr = static_pointer_cast<MxpiTensorPackageList>(metadata);
    std::vector <RotatedObjectInfo> rObjInfos;
    std::vector <RotatedObjectInfo> results;
    auto mxpiRotateobjListptr = std::make_shared<mxpirotateobjproto::MxpiRotateobjProtoList>();   
    GenerateMxpiRotateobjList(rObjInfos, results, visionInfos, 
                              *srcMxpiTensorPackageListSptr, *mxpiRotateobjListptr);   
    std::string rotateObjProtoName = "mxpi_rotateobjproto";
    APP_ERROR ret = mxpiMetadataManager.AddProtoMetadata(rotateObjProtoName, 
                                                         static_pointer_cast<void>(mxpiRotateobjListptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, rotateObjProtoName) << "MxpiRotateObjPostProcess add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, rotateObjProtoName, mxpiErrorInfo);
        return ret;
    }
    SendData(0, *buffer);
    LogInfo << "MxpiRotateObjPostProcess::Process end";
    return APP_ERR_OK;   
}

/**
* @brief Definition the parameter of configure properties.
* @return std::vector<std::shared_ptr<void>>
*/
std::vector<std::shared_ptr<void>> MxpiRotateObjPostProcess::DefineProperties(){
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto imageResizeNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
        STRING, "imageSource", "inputName", "the name of imageresize", "mxpi_imageresize0", "NULL", "NULL"});
    properties.push_back(imageResizeNameProSptr);
    return properties;
}

}

// Register the MxpiRotateObjPostProcess plugin through macro
MX_PLUGIN_GENERATE(MxpiRotateObjPostProcess)
