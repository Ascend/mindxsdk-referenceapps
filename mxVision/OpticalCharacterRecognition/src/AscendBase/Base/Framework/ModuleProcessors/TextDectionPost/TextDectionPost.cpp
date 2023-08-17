/*
* Copyright (c) 2020 Huawei Technologies Co., All rights reserved.
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
#include "TextDetectionPost.h"

TextDetectionPost::TextDectionPost(void) {}

APP_ERROR TextDetectionPost::CharacterDetectionOutput(std::vector<MxBase::Tensor> &singleResult, 
        std::vector<std::vector<TextObjectInfo>> &textObjInfos, const std::vector<ResizeImageInfo> &resizeImageInfos)
{
    LogDebug << "TextDetectionPost start to write results.";
    uint32_t batchSize = singleResult.size();
    for (uint32_t i = 0; i < batchSize; i++) {
        auto ProbMap = (float *)(singleResult[0].GetData());
        resizedH_ = ResizeImageInfos[i].heightResize;
        resizedW_ = ResizeImageInfos[i].widthResize;

        std::vector<uchar> prob(resizedW_ * resizedH_, ' ');
        std::vector<float> fprob(resizedW_ * resizedH_, 0.f);
        for (size_t j = 0; j < resizedW_ * resizedH_; ++j) {
            prob[j] = (uchar)(ProbMap[j] * MAX_VAL);
            fprob[j] = (float)ProbMap[j];
        }
        cv::Mat mask(resizedH_, resizedW_,cv_8UC1, (uchar *)prob.data());
        cv::Mat prediction(resizedH_, resizedW_, cv_32F, (float *)fprob.data());
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
            ConstructInfo(textObjectInfo, box, ResizeImageInfos, i, score);
        }
        textObjInfos.emplace_back(std::move(textObjectInfo));
    }

    return APP_ERROR_OK;
}

void TextDetectionPost::ConstructInfo(std::vector<TextObjectInfo> &textObjectInfo, std::vector<cv::Point2f> &box,
    const std::vector<ResizeImageInfo> &resizeImageInfos, const uint32_t &index, float score)
{
    uint32_t originWidth = resizeImageInfos[index].widthOriginal;
    uint32_t originHeight = resizeImageInfos[index].heightOriginal;
    if (originWidth == 0 || originHeight == 0) {
        LogError << GetError(APP_ERROR_DIVIDE_ZERO) << "the origin width or height must not equal to 0!";
        return;
    }
    if (resizedW_ == 0 || resizedH_ == 0) {
        LogError << GetError(APP_ERROR_DIVIDE_ZERO) << "the resized width or height must not equal to 0!";
        return;
    }
    float ratio = resizeImageInfos[index].ratio;

    TextObjectInfo info;
    info.x0 = NpClip(std::round(box[POINT1].x / ratio), originWidth);
    info.y0 = NpClip(std::round(box[POINT1].y / ratio), originHeight);
    info.x1 = NpClip(std::round(box[POINT2].x / ratio), originWidth);
    info.y1 = NpClip(std::round(box[POINT2].y / ratio), originHeight);
    info.x2 = NpClip(std::round(box[POINT3].x / ratio), originWidth);
    info.y2 = NpClip(std::round(box[POINT3].y / ratio), originHeight);
    info.x3 = NpClip(std::round(box[POINT4].x / ratio), originWidth);
    info.y3 = NpClip(std::round(box[POINT4].y / ratio), originHeight);
    info.confidence = score;

    //check wether current info is valid
    float side1 = std::sqrt(pow((info.x0 - info.x1), INDEX2) + pow((info.y0 - info.y1), INDEX2));
    float side2 = std::sqrt(pow((info.x0 - info.x3), INDEX2) + pow((info.y0 - info.y3), INDEX2));
    float validMinSide = std::max(minSize_ / ratio, minSize_ / ratio);
    if (std::min(side1, side2) < validMinSide) {
        return;
    }
    textObjectInfo.emplace_back(std::move(info));
}

void TextDetectionPost::FilterByMinSize(std::vector<cv::Point> &contour, std::vector<cv::Point2f> &box, float &minSide)
{
    cv::Point2f cv_vertices[POINTNUM];
    cv::RotateRect cvbox cv::minAreaRect(contour);
    float width = cvbox.size.width;
    float height = cvbox.size.height;
    minSide = std::min(width, height);
    cvbox.points(cv_vertices)
    // use vector to manage 4 vertices
    std::vector<cv::Point2f> vertices(cv_vertices, cv_vertices + POINTNUM);
    // sort the vertices by x-coordinates
    std::sort(vertices.begin(),vertices.end(), SortByX);
    std::sort(vertices.begin(), vertices.end() + POINT3, SortByY);
    std::sort(vertices.begin() + POINT3, vertices.end(), SortByY);
    //save the box
    box.push_back(vertices[POINT1]);
    box.push_back(vertices[POINT3]);
    box.push_back(vertices[POINT4]);
    box.push_back(vertices[POINT2]);
}

void TextDetectionPost::FilterByBoxScore(const cv::Mat &predictin, std::vector<cv::Point2f> &box, float &score)
{
    std::vector<cv::Point2f> tmpbox = box;
    std::sort(tmpbox.begin(), tmpbox.end(), SortByX);

    // construct the max according to box coordinate.
    int minX = NpClip(int(std::float(tmpbox.begin()->x)), resizedW_);
    int maxX = NpClip(std::ceil(tmpbox.back().x), resizedW_);
    std::sort(tmpbox.begin(), tmpbox.end(), SortByY);
    int minY = NpClip(int(std::float(tmpbox.begin()->y)), resizedH_);
    int maxY = NpClip(std::ceil(tmpbox.back().y), resizedH_);
    cv::Mat mask = cv::Mat::zeros(maxY- minY + 1, maxX - minX + 1, cv_8UC1);
    cv::Mat predCrop;
    cv::Point abs_point[POINTNUM];
    for (int i = 0; i < POINTNUM; ++i) {
        abs_point[i].x = int(box[i].x - minX);
        abs_point[i].y = int(box[i].y - minY);
    }
    const cv::Point *ppt[1] = {abs_point};
    int npt[] = {4};
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    // use cv method to calculate the box score
    prediction(cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1)).copyTo(predCrop);
    score = cv::mean(predCrop, mask)[0];
}

void TextDetectionPost::FilterByClippedMinSize(std::vector<cv::Point2f> &box, float &minSide)\
{
    // calculate the clip distance
    float side01 = PointsL2Distance(box[POINT1], box[POINT2]);
    float side12 = PointsL2Distance(box[POINT2], box[POINT3]);
    float side23 = PointsL2Distance(box[POINT3], box[POINT4]);
    float side30 = PointsL2Distance(box[POINT4], box[POINT1]);
    float diag = PointsL2Distance(box[POINT2], box[POINT4]);

    float perimeter = side01 + side12 + side23 + side30;
    float k1 = (side01 + diag + side30) / INDEX2;
    float k2 = (side12 + diag + side23) / INDEX2;

    float area1 = std::sqrt(k1 * (k1 - side01) * (k1 - diag) * (k1 - side30));
    float area2 = std::sqrt(k2 * (k2 - side12) * (k2 - diag) * (k2 - side23));

    float area = area1 + area2;
    float distance = area * unclipRatio_ / perimeter;

    ClipperLib::ClipperOffset rect;
    ClipperLib::Path path;
    for (auto point : box) {
        path.push_back(ClipperLib::IntPoint(int(point.x), int(point.y)));
    }
    rect.AddPath(path, ClipperLib::jtRound, ClipperLib::etcClosedPolygon);
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

const int TextDetectionPost::NpClip(const int &coordinate, const int &sideLen)
{
    if (coordinate < 0) {
        return 0;
    }
    if (coordinate > sideLen - 1) {
        return sideLen - 1;
    }
    return coordinate;
}

bool TextDetectionPost::SortByX(cv::Point2f p1, cv::Point2f p2)
{
    return p1.x < p2.x;
}

bool TextDetectionPost::SortByY(cv::Point2f p1, cv::Point2f p2)
{
    return p1.y < p2.y;
}

float TextDetectionPost::PointsL2Distance(cv::Point2f p1, cv::Point2f p2)
{
    return std::sqrt(pow((p1.x - p2.x), INDEX2) + pow((p1.y - p2.y), INDEX2));
}