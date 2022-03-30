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

#include <cmath>
#include <algorithm>
#include <iostream>
#include "clipper.hpp"
#include "DBPostProcess.h"

namespace MxBase {
DBPostProcess &DBPostProcess::operator = (const DBPostProcess &other)
{
    if (this == &other) {
        return *this;
    }
    TextObjectPostProcessBase::operator = (other);
    minSize_ = other.minSize_;
    thresh_ = other.thresh_;
    boxThresh_ = other.boxThresh_;
    unclipRatio_ = other.unclipRatio_;
    resizedH_ = other.resizedH_;
    resizedW_ = other.resizedW_;
    candidates_ = other.candidates_;
    return *this;
}

DBPostProcess::DBPostProcess(const DBPostProcess &other)
{
    minSize_ = other.minSize_;
    thresh_ = other.thresh_;
    boxThresh_ = other.boxThresh_;
    unclipRatio_ = other.unclipRatio_;
    resizedH_ = other.resizedH_;
    resizedW_ = other.resizedW_;
    candidates_ = other.candidates_;
}

/*
 * @description Load the configs and labels from the file.
 * @param labelPath config path and label path.
 * @return APP_ERROR error code.
 */
APP_ERROR DBPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig)
{
    LogInfo << "Begin to initialize DBPostProcess.";
    // Open config file
    APP_ERROR ret = TextObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in DBPostProcess.";
        return ret;
    }
    ret = configData_.GetFileValue<int>("MIN_SIZE", minSize_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read MIN_SIZE from config, default is: " << minSize_;
    }
    ret = configData_.GetFileValue<float>("THRESH", thresh_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read THRESH from config, default is: " << thresh_;
    }
    ret = configData_.GetFileValue<float>("BOX_THRESH", boxThresh_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read BOX_THRESH from config, default is: " << boxThresh_;
    }
    ret = configData_.GetFileValue<float>("UNCLIP_RATIO", unclipRatio_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read unclip ratio from config, default is: " << unclipRatio_;
    }
    ret = configData_.GetFileValue<int>("CANDIDATES", candidates_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "Fail to read candidates from config, default is: " << candidates_;
    }
    LogInfo << "End to initialize DBPostProcess.";
    return APP_ERR_OK;
}

/*
 * @description: Do nothing temporarily.
 * @return: APP_ERROR error code.
 */
APP_ERROR DBPostProcess::DeInit()
{
    LogInfo << "Begin to deinitialize DBPostProcess.";
    LogInfo << "End to deialize DBPostProcess.";
    return APP_ERR_OK;
}

APP_ERROR DBPostProcess::Process(const std::vector<TensorBase> &tensors,
    std::vector<std::vector<TextObjectInfo>> &textObjInfos, const std::vector<ResizedImageInfo> &resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogDebug << "Start to process DBPostProcess.";
    auto inputs = tensors;
    APP_ERROR ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "CheckAndMoveTensors failed.";
        return ret;
    }
    ret = CheckResizeType(resizedImageInfos);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Unsupported resize type for current postprocess lib.";
        return ret;
    }
    ObjectDetectionOutput(inputs, textObjInfos, resizedImageInfos);
    LogDebug << "End to process DBPostProcess.";
    return APP_ERR_OK;
}

void DBPostProcess::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
    std::vector<std::vector<TextObjectInfo>> &textObjInfos, const std::vector<ResizedImageInfo> &resizedImageInfos)
{
    LogDebug << "DBPostProcess start to write results.";
    auto shape = tensors[0].GetShape();
    uint32_t batchSize = shape[0];
    for (uint32_t i = 0; i < batchSize; i++) {
        auto ProbMap = (float *)GetBuffer(tensors[0], i);
        resizedH_ = resizedImageInfos[i].heightResize;
        resizedW_ = resizedImageInfos[i].widthResize;

        std::vector<uchar> prob(resizedW_ * resizedH_, ' ');
        std::vector<float> fprob(resizedW_ * resizedH_, 0.f);
        for (size_t j = 0; j < resizedW_ * resizedH_; ++j) {
            prob[j] = (uchar)(ProbMap[j] * MAX_VAL);
            fprob[j] = (float)ProbMap[j];
        }
        cv::Mat mask(resizedH_, resizedW_, CV_8UC1, (uchar *)prob.data());
        cv::Mat prediction(resizedH_, resizedW_, CV_32F, (float *)fprob.data());
        cv::Mat binmask;

        cv::threshold(mask, binmask, (float)(thresh_ * MAX_VAL), MAX_VAL, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binmask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        int contourNum = NpClip(contours.size(), candidates_);
        std::vector<TextObjectInfo> textObjectInfo;

        // traverse and filter all contours
        for (int j = 0; j < contourNum; j++) {
            std::vector<cv::Point> contour = contours[j];
            std::vector<cv::Point2f> box;
            float minSide1 = 0.f;
            float minSide2 = 0.f;
            float score = 0.f;
            // 1st filter
            FilterByMinSize(contour, box, minSide1);
            if (minSide1 < minSize_) {
                continue;
            }
            // 2nd filter
            FilterByBoxScore(prediction, box, score);
            if (score < boxThresh_) {
                continue;
            }
            // 3rd filter
            FilterByClippedMinSize(box, minSide2);
            if (minSide2 < minSize_ + UNCLIP_DISTANCE) {
                continue;
            }
            // write box info into TextObjectInfo
            ConstructInfo(textObjectInfo, box, resizedImageInfos, i, score);
        }
        textObjInfos.emplace_back(std::move(textObjectInfo));
    }
}

void DBPostProcess::ConstructInfo(std::vector<TextObjectInfo> &textObjectInfo, std::vector<cv::Point2f> &box,
    const std::vector<ResizedImageInfo> &resizedImageInfos, const uint32_t &index, float score)
{
    uint32_t originWidth = resizedImageInfos[index].widthOriginal;
    uint32_t originHeight = resizedImageInfos[index].heightOriginal;
    if (originWidth == 0 || originHeight == 0) {
        LogError << GetError(APP_ERR_DIVIDE_ZERO) << "the origin width or height must not equal to 0!";
        return;
    }
    if (resizedW_ == 0 || resizedH_ == 0) {
        LogError << GetError(APP_ERR_DIVIDE_ZERO) << "the resized width or height must not equal to 0!";
        return;
    }
    float ratioX = std::min(resizedW_ / (float)originWidth, resizedH_ / (float)originHeight);
    float ratioY = ratioX;
    ResizeType resizeType = resizedImageInfos[index].resizeType;
    if (resizeType == RESIZER_STRETCHING) {
        ratioX = resizedW_ / originWidth;
        ratioY = resizedH_ / originHeight;
    }
    if (ratioX == 0 || ratioY == 0) {
        LogError << GetError(APP_ERR_DIVIDE_ZERO) << "the ratio of width or height must not equal to 0!";
        return;
    }
    TextObjectInfo info;
    info.x0 = NpClip(std::round(box[POINT1].x / ratioX), originWidth);
    info.y0 = NpClip(std::round(box[POINT1].y / ratioY), originHeight);
    info.x1 = NpClip(std::round(box[POINT2].x / ratioX), originWidth);
    info.y1 = NpClip(std::round(box[POINT2].y / ratioY), originHeight);
    info.x2 = NpClip(std::round(box[POINT3].x / ratioX), originWidth);
    info.y2 = NpClip(std::round(box[POINT3].y / ratioY), originHeight);
    info.x3 = NpClip(std::round(box[POINT4].x / ratioX), originWidth);
    info.y3 = NpClip(std::round(box[POINT4].y / ratioY), originHeight);
    info.confidence = score;

    // check whether current info is valid
    float side1 = std::sqrt(pow((info.x0 - info.x1), INDEX2) + pow((info.y0 - info.y1), INDEX2));
    float side2 = std::sqrt(pow((info.x0 - info.x3), INDEX2) + pow((info.y0 - info.y3), INDEX2));
    float validMinSide = std::max(minSize_ / ratioX, minSize_ / ratioY);
    if (std::min(side1, side2) < validMinSide) {
        return;
    }
    textObjectInfo.emplace_back(std::move(info));
}

void DBPostProcess::FilterByMinSize(std::vector<cv::Point> &contour, std::vector<cv::Point2f> &box, float &minSide)
{
    cv::Point2f cv_vertices[POINTNUM];
    cv::RotatedRect cvbox = cv::minAreaRect(contour);
    float width = cvbox.size.width;
    float height = cvbox.size.height;
    minSide = std::min(width, height);
    cvbox.points(cv_vertices);
    // use vector to manage 4 vertices
    std::vector<cv::Point2f> vertices(cv_vertices, cv_vertices + POINTNUM);
    // sort the vertices by x-coordinates
    std::sort(vertices.begin(), vertices.end(), SortByX);
    std::sort(vertices.begin(), vertices.begin() + POINT3, SortByY);
    std::sort(vertices.begin() + POINT3, vertices.end(), SortByY);
    // save the box
    box.push_back(vertices[POINT1]);
    box.push_back(vertices[POINT3]);
    box.push_back(vertices[POINT4]);
    box.push_back(vertices[POINT2]);
}

void DBPostProcess::FilterByBoxScore(const cv::Mat &prediction, std::vector<cv::Point2f> &box, float &score)
{
    std::vector<cv::Point2f> tmpbox = box;
    std::sort(tmpbox.begin(), tmpbox.end(), SortByX);

    // construct the mask according to box coordinates.
    int minX = NpClip(int(std::floor(tmpbox.begin()->x)), resizedW_);
    int maxX = NpClip(std::ceil(tmpbox.back().x), resizedW_);
    std::sort(tmpbox.begin(), tmpbox.end(), SortByY);
    int minY = NpClip(int(std::floor(tmpbox.begin()->y)), resizedH_);
    int maxY = NpClip(int(std::ceil(tmpbox.back().y)), resizedH_);
    cv::Mat mask = cv::Mat::zeros(maxY - minY + 1, maxX - minX + 1, CV_8UC1);
    cv::Mat predCrop;
    cv::Point abs_point[POINTNUM];
    for (int i = 0; i < POINTNUM; ++i) {
        abs_point[i].x = int(box[i].x - minX);
        abs_point[i].y = int(box[i].y - minY);
    }
    const cv::Point* ppt[1] = {abs_point};
    int npt[] = {4};
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    // use cv method to calculate the box score
    prediction(cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1)).copyTo(predCrop);
    score = cv::mean(predCrop, mask)[0];
}

void DBPostProcess::FilterByClippedMinSize(std::vector<cv::Point2f> &box, float &minSide)
{
    // calculate the clip distance
    float side01 = PointsL2Distance(box[POINT1], box[POINT2]);
    float side12 = PointsL2Distance(box[POINT2], box[POINT3]);
    float side23 = PointsL2Distance(box[POINT3], box[POINT4]);
    float side30 = PointsL2Distance(box[POINT4], box[POINT1]);
    float diag = PointsL2Distance(box[POINT2], box[POINT4]);

    float perimeter = side01 + side12 + side23 + side30;
    float k1 = (side01 + diag + side30) / INDEX2;
    float k2 = (side12 + side23 + diag) / INDEX2;
    float area1 = std::sqrt(k1 * (k1 - side01) * (k1 - diag) * (k1 - side30));
    float area2 = std::sqrt(k2 * (k2 - side12) * (k2 - side23) * (k2 - diag));

    float area = area1 + area2;
    float distance = area * unclipRatio_ / perimeter;

    ClipperLib::ClipperOffset rect;
    ClipperLib::Path path;
    for (auto point : box) {
        path.push_back(ClipperLib::IntPoint(int(point.x), int(point.y)));
    }
    rect.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
    ClipperLib::Paths result;
    rect.Execute(result, distance);

    std::vector<cv::Point> contour;
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[result.size() - 1].size(); ++j) {
            contour.emplace_back(result[i][j].X, result[i][j].Y);
        }
    }
    // check for exception
    box.clear();
    FilterByMinSize(contour, box, minSide);
}

const int DBPostProcess::NpClip(const int &coordinate, const int &sideLen)
{
    if (coordinate < 0) {
        return 0;
    }
    if (coordinate > sideLen - 1) {
        return sideLen - 1;
    }
    return coordinate;
}

bool DBPostProcess::SortByX(cv::Point2f p1, cv::Point2f p2)
{
    return p1.x < p2.x;
}

bool DBPostProcess::SortByY(cv::Point2f p1, cv::Point2f p2)
{
    return p1.y < p2.y;
}

APP_ERROR DBPostProcess::CheckResizeType(const std::vector<ResizedImageInfo> &resizedImageInfos)
{
    for (auto info : resizedImageInfos) {
        if (info.resizeType != RESIZER_STRETCHING && info.resizeType != RESIZER_MS_KEEP_ASPECT_RATIO) {
            return APP_ERR_COMM_INVALID_PARAM;
        }
    }
    return APP_ERR_OK;
}

float DBPostProcess::PointsL2Distance(cv::Point2f p1, cv::Point2f p2)
{
    return std::sqrt(pow((p1.x - p2.x), INDEX2) + pow((p1.y - p2.y), INDEX2));
}

bool DBPostProcess::IsValidTensors(const std::vector<TensorBase> &tensors) const
{
    auto shape = tensors[0].GetShape();
    if (shape.size() != VECTOR_FIFTH_INDEX) {
        LogError << "number of tensor dimensions (" << shape.size() << ") "
                 << "is not equal to (" << VECTOR_FIFTH_INDEX << ")";
        return false;
    }
    return true;
}
#ifdef ENABLE_POST_PROCESS_INSTANCE
extern "C" {
std::shared_ptr<MxBase::DBPostProcess> GetTextObjectInstance()
{
    LogInfo << "Begin to get DBPostProcess instance.";
    auto instance = std::make_shared<DBPostProcess>();
    LogInfo << "End to get DBPostProcess instance.";
    return instance;
}
}
#endif
}