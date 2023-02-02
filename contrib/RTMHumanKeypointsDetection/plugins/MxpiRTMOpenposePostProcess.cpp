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
#include "MxpiRTMOpenposePostProcess.h"
#include <numeric>
#include <algorithm>
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"

using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
using namespace cv;

namespace {
    auto g_uint8Deleter = [] (uint8_t *p) { };
    const int K_NUM_BODY_PARTS = 18;
    const int K_UPSAMPLED_STRIDE = 8;

    // CocoSkeletonsNetwork
    const std::vector<unsigned int> K_POSE_MAP_INDEX {
        12, 13, 20, 21, 14, 15, 16, 17, 22, 23,
        24, 25, 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 28, 29, 30, 31, 34, 35,
        32, 33, 36, 37, 18, 19, 26, 27
    };

    // CocoSkeletons
    const std::vector<unsigned int> K_POSE_BODY_PART_SKELETONS {
        1, 2, 1, 5, 2, 3, 3, 4, 5, 6,
        6, 7, 1, 8, 8, 9, 9, 10, 1, 11,
        11, 12, 12, 13, 1, 0, 0, 14, 14, 16,
        0, 15, 15, 17, 2, 16, 5, 17
    };
    // Nms score threshold
    const float K_NMS_THRESHOLD = 0.1;
    // Range of nearest neighbors
    const int K_NEAREST_KEYPOINTS_THRESHOLD = 6;
    // PAF score threshold as a valid inner point on a skeleton
    const float K_LOCAL_PAF_SCORE_THRESHOLD = 0.05;
    // The minimum number of valid inner points a skeleton includes to be regarded as a correct skeleton
    const int K_LOCAL_PAF_COUNT_THRESHOLD = 8;
    // The minimum number of skeletons needed to form a person
    const int K_PERSON_SKELETON_COUNT_THRESHOLD = 3;
    // The lowest average score per keypoint in a person
    const float K_PERSON_KEYPOINT_AVG_SCORE_THRESHOLD = 0.2;
    // RGB value of green
    const int RGB_GREEN_VALUE = 85;
    // RGB value of red
    const int RGB_RED_VALUE = 255;
    // RGB value of blue
    const int RGB_BLUE_VALUE = 0;
    // the thickness of lines
    const int THICKNESS_OF_LINES = 3;
    // the linetype of lines
    const int LINETYPE_OF_LINES = 8;
    // the radiu of dots
    const int RADIU_OF_DOTS = 3;
}

/**
 * @brief decode MxpiTensorPackageList
 * @param tensorPackageList - Source tensorPackageList
 * @param tensors - Target TensorBase data
 */
static void GetTensors(const MxTools::MxpiTensorPackageList tensorPackageList,
                       std::vector<MxBase::TensorBase> &tensors)
{
    for (int i = 0; i < tensorPackageList.tensorpackagevec_size(); ++i) {
        for (int j = 0; j < tensorPackageList.tensorpackagevec(i).tensorvec_size(); j++) {
            MxBase::MemoryData memoryData = {};
            memoryData.deviceId = tensorPackageList.tensorpackagevec(i).tensorvec(j).deviceid();
            memoryData.type = (MxBase::MemoryData::MemoryType)tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).memtype();
            memoryData.size = (uint32_t)tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordatasize();
            memoryData.ptrData = (void *)tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordataptr();
            std::vector<uint32_t> outputShape = {};
            for (int k = 0; k < tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensorshape_size(); ++k) {
                outputShape.push_back((uint32_t)tensorPackageList.
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
 * @breif Get infomation of origin input image and aligned image
 * @param visionList - MxpiVisionList object obtained from upstream plugin
 * @param visionInfos - Result vector with elements [originHeight, originWidth, aligned_height, aligned_width]
 */
static void GetImageSizes(const MxTools::MxpiVisionList visionList, std::vector<int> &visionInfos)
{
    MxpiVision vision = visionList.visionvec(0);
    MxpiVisionInfo visionInfo = vision.visioninfo();
    visionInfos.push_back(visionInfo.height());
    visionInfos.push_back(visionInfo.width());
    visionInfos.push_back(visionInfo.heightaligned());
    visionInfos.push_back(visionInfo.widthaligned());
}

/**
 * @brief Parsing TensorBase data to keypoint heatmap and PAF heatmap of openpose model
 * @param tensors - TensorBase vector
 * @return Two-element vector, keeping keypoint heatmap and paf heatmap respectively
 */
static std::vector<std::vector<cv::Mat> > ReadDataFromTensorPytorch(const std::vector <MxBase::TensorBase> &tensors)
{
    const int heightIndex = 2, widthIndex = 3;
    auto shape = tensors[2].GetShape();
    int channelKeypoint = shape[1];
    int height = shape[heightIndex];
    int width = shape[widthIndex];
    auto shapeP = tensors[3].GetShape();
    int channelPaf = shapeP[1];
    // Read keypoint data
    auto dataPtr = (uint8_t *)tensors[2].GetBuffer();
    std::shared_ptr<void> keypointPointer;
    keypointPointer.reset(dataPtr, g_uint8Deleter);
    std::vector<cv::Mat> keypointHeatmap {};
    int idx = 0;
    for (int i = 0; i < channelKeypoint; i++) {
        cv::Mat singleChannelMat(height, width, CV_32FC1, cv::Scalar(0));
        for (int j = 0; j < height; j++) {
            float *ptr = singleChannelMat.ptr<float>(j);
            for (int k = 0; k < width;  k++) {
                ptr[k] = static_cast<float *>(keypointPointer.get())[idx];
                idx += 1;
            }
        }
        keypointHeatmap.push_back(singleChannelMat);
    }
    // Read PAF data
    auto dataPafPtr = (uint8_t *)tensors[3].GetBuffer();
    std::shared_ptr<void> pafPointer;
    pafPointer.reset(dataPafPtr, g_uint8Deleter);
    std::vector<cv::Mat> pafHeatmap {};
    idx = 0;
    for (int i = 0; i < channelPaf; i++) {
        cv::Mat singleChannelMat(height, width, CV_32FC1, cv::Scalar(0));
        for (int j = 0; j < height; j++) {
            float *ptr = singleChannelMat.ptr<float>(j);
            for (int k = 0; k < width;  k++) {
                ptr[k] = static_cast<float *>(pafPointer.get())[idx];
                idx += 1;
            }
        }
        pafHeatmap.push_back(singleChannelMat);
    }
    std::vector<std::vector<cv::Mat> > result = {keypointHeatmap, pafHeatmap};
    return result;
}

/**
 * @brief Non-Maximum Suppression, keep points that is greater than all its four surround points,
 * i.e. up, bottom, left and right points
 * @param plain - 2D data for NMS
 * @param threshold - NMS threshold
 */
static void NMS(cv::Mat &plain, float threshold)
{
    // Keep points with score below the NMS score threshold are set to 0
    plain.setTo(0, plain < threshold);
    // Find points that is greater than all its four surround points
    cv::Mat plainWithBorder;
    const int borderPadding = 2;
    const int bottomRightIndex = 2;
    cv::copyMakeBorder(plain, plainWithBorder, borderPadding, borderPadding, borderPadding, borderPadding,
                       BORDER_CONSTANT, cv::Scalar(0));
    cv::Mat plainWithBorderClone = plainWithBorder.clone();
    int subMatCols = plainWithBorder.cols - borderPadding;
    int subMatRows = plainWithBorder.rows - borderPadding;
    cv::Mat plainCenter = plainWithBorder(cv::Rect(1, 1, subMatCols, subMatRows));
    cv::Mat plainBottom = plainWithBorder(cv::Rect(1, bottomRightIndex, subMatCols, subMatRows));
    cv::Mat plainUp = plainWithBorder(cv::Rect(1, 0, subMatCols, subMatRows));
    cv::Mat plainLeft = plainWithBorder(cv::Rect(0, 1, subMatCols, subMatRows));
    cv::Mat plainRight = plainWithBorder(cv::Rect(bottomRightIndex, 1, subMatCols, subMatRows));
    int count = 0;
    for (int i = 0; i < plainCenter.rows; i++) {
        float *centerPtr = plainCenter.ptr<float>(i);
        float *bottomPtr = plainBottom.ptr<float>(i);
        float *upPtr = plainUp.ptr<float>(i);
        float *leftPtr = plainLeft.ptr<float>(i);
        float *rightPtr = plainRight.ptr<float>(i);
        float *cloneBorderPtr = plainWithBorderClone.ptr<float>(i + 1);
        for (int j = 0; j < plainCenter.cols; j++) {
            if (!((centerPtr[j] > upPtr[j]) && (centerPtr[j] > bottomPtr[j]) &&
                (centerPtr[j] > leftPtr[j]) && (centerPtr[j] > rightPtr[j]))) {
                cloneBorderPtr[j + 1] = 0;
            }
        }
    }
    plain = plainWithBorderClone(cv::Rect(borderPadding, borderPadding, plainCenter.cols - borderPadding,
                                          plainCenter.rows - borderPadding)).clone();
}

/**
 * @brief Comparation between two PartPair elements
 * @param p1 - PartPair p1
 * @param p2 - PartPair p2
 * @return True if the score of p1 is greater than that of p2
 */
static bool GreaterSort(PartPair p1, PartPair p2)
{
    return p1.score > p2.score;
}

/**
 * @brief Comparation between two cv::Point elements
 * @param p1 - cv::Point p1
 * @param p2 - cv::Point p2
 * @return True if the x coordinate of p2 is greater than that of p1
 */
static bool PointSort(cv::Point p1, cv::Point p2)
{
    return p1.x < p2.x;
}

/**
 * @brief Resize output heatmaps to the size of the origin image
 * @param keypointHeatmap - Keypoint heatmap, each channel of the heatmap is stored as a Mat
 * @param pafHeatmap - PAF heatmap, each channel of the heatmap is stored as a Mat
 * @param visionInfos - Vision infos of origin image and aligned image
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::ResizeHeatmaps(std::vector<cv::Mat> &keypointHeatmap,
    std::vector<cv::Mat > &pafHeatmap,
    std::vector<int> &visionInfos)
{
    // Calculate padding direction and padding value
    int originHeight = visionInfos[0];
    int originWidth = visionInfos[1];
    // padding along height
    int paddingDirection = 0;
    if (originHeight > originWidth) {
        // padding along width
        paddingDirection = 1;
    }
    int paddingValue = 0;
    if (paddingDirection == 0) {
        // pad height
        paddingValue = floor(inputHeight_ - inputWidth_ * originHeight / originWidth);
    } else {
        // pad width
        paddingValue = floor(inputWidth_ - inputHeight_ * originWidth / originHeight);
    }
    // Channel Split Resize
    int height = keypointHeatmap[0].rows;
    int width = keypointHeatmap[0].cols;
    // use opencv parallel_for_ for parallel computing
    parallel_for_(Range(0, keypointHeatmap.size()), [&](const Range& r)
    {
        for (int i = r.start; i < r.end; i++) {
            cv::Mat singleChannelMat = keypointHeatmap[i];
            cv::resize(singleChannelMat, singleChannelMat, Size(0, 0),
                    K_UPSAMPLED_STRIDE, K_UPSAMPLED_STRIDE, INTER_CUBIC);
            if (paddingDirection == 0) {
                // remove height padding
                singleChannelMat =
                        singleChannelMat(cv::Rect(0, 0, singleChannelMat.cols, singleChannelMat.rows - paddingValue));
            } else {
                // remove width padding
                singleChannelMat =
                        singleChannelMat(cv::Rect(0, 0, singleChannelMat.cols - paddingValue, singleChannelMat.rows));
            }
            keypointHeatmap[i] = singleChannelMat;
        }
    });
    parallel_for_(Range(0, pafHeatmap.size()), [&](const Range& r)
    {
        for (int i = r.start; i < r.end; i++) {
            cv::Mat singleChannelMat = pafHeatmap[i];
            cv::resize(singleChannelMat, singleChannelMat, Size(0, 0),
                    K_UPSAMPLED_STRIDE, K_UPSAMPLED_STRIDE, INTER_CUBIC);
            if (paddingDirection == 0) {
                singleChannelMat =
                        singleChannelMat(cv::Rect(0, 0, singleChannelMat.cols, singleChannelMat.rows - paddingValue));
            } else {
                singleChannelMat =
                        singleChannelMat(cv::Rect(0, 0, singleChannelMat.cols - paddingValue, singleChannelMat.rows));
            }
            pafHeatmap[i] = singleChannelMat;
        }
    });
    return APP_ERR_OK;
}

/**
 * @brief Extract candidate keypoints
 * @param keypointHeatmap - Resized keypoint heatmap
 * @param coor - Keep extracted result, store a point in a cv::Point object,
 * store keypoints of different channel in different vectors
 * @param coorScore - Scores corresponding to extracted keypoints
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::ExtractKeypoints(std::vector<cv::Mat> &keypointHeatmap,
    std::vector<std::vector<cv::Point> > &coor,
    std::vector<std::vector<float> > &coorScore)
{
    const int polynomialExponent = 2;
    int keypointHeatmap_size =  keypointHeatmap.size() - 1;
    for (int i = 0; i < keypointHeatmap_size; i++) {
        // NMS
        NMS(keypointHeatmap[i], K_NMS_THRESHOLD);
        std::vector<cv::Point> nonZeroCoordinates;
        cv::findNonZero(keypointHeatmap[i], nonZeroCoordinates);
        std::sort(nonZeroCoordinates.begin(), nonZeroCoordinates.end(), PointSort);
        std::vector<int> suppressed(nonZeroCoordinates.size(), 0);
        std::vector<cv::Point> keypointsWithoutNearest {};
        std::vector<float> keypointsScore {};
        // Remove other keypoints within a certain range around one keypoints
        int nonZeroCoordinates_size =  nonZeroCoordinates.size();
        for (int j = 0; j < nonZeroCoordinates_size; j++) {
            if (suppressed[j]) {
                continue;
            }
            int thrownIndex = j + 1;
            auto it_end = std::end(nonZeroCoordinates);
            for (auto it = std::begin(nonZeroCoordinates) + j + 1; it != it_end; it++) {
                if ((it->x - nonZeroCoordinates[j].x) > K_NEAREST_KEYPOINTS_THRESHOLD) {
                    break;
                }
                float distance = (nonZeroCoordinates[j].x - it->x) * (nonZeroCoordinates[j].x - it->x) +
                (nonZeroCoordinates[j].y - it->y) * (nonZeroCoordinates[j].y - it->y);
                if (distance < K_NEAREST_KEYPOINTS_THRESHOLD * K_NEAREST_KEYPOINTS_THRESHOLD) {
                    thrownIndex = std::distance(std::begin(nonZeroCoordinates) + thrownIndex, it) + thrownIndex;
                    suppressed[thrownIndex] = 1;
                }
            }
            
            keypointsWithoutNearest.push_back(nonZeroCoordinates[j]);
            keypointsScore.push_back(keypointHeatmap[i].at<float>(
                nonZeroCoordinates[j].y, nonZeroCoordinates[j].x));
        }
        coor.push_back(keypointsWithoutNearest);
        coorScore.push_back(keypointsScore);
    }
    return APP_ERR_OK;
}

/**
 * @brief Compute expected confidence for each candidate skeleton
 * @param endpoints - Coordinates of the two end points of a skeleton
 * @param pafX - PAF heatmap of x coordinate
 * @param pafY - PAF heatmap of y coordinate
 * @return result - Keep confidence information of this skeleton in the form:
 * [confidence score, number of successfully hit sub points]
 */
std::vector<float> MxpiRTMOpenposePostProcess::OneSkeletonScore(std::vector<cv::Point> &endpoints,
    const cv::Mat &pafX, const cv::Mat &pafY)
{
    int x1 = endpoints[0].x, y1 = endpoints[0].y;
    int x2 = endpoints[1].x, y2 = endpoints[1].y;
    // affinity score of this skeleton
    float score = 0;
    // count: number of valid inner points on this skeleton
    int count = 0, numInter = 10;
    float dx = x2 - x1;
    float dy = y2 - y1;
    float normVec = sqrt(dx * dx + dy * dy);
    float vx = dx / (normVec + 1e-6);
    float vy = dy / (normVec + 1e-6);
    // generate 10 points equally spaced on this skeleton
    std::vector<int> xs {};
    float stepX = dx / (numInter - 1);
    float tempX = 0;
    for (int k = 0; k < numInter; k++) {
        tempX = x1 + k * stepX;
        xs.push_back(round(tempX));
    }
    std::vector<int> ys {};
    float stepY = dy / (numInter - 1);
    float tempY = 0;
    for (int k = 0; k < numInter; k++) {
        tempY = y1 + k * stepY;
        ys.push_back(round(tempY));
    }
    std::vector<float> subScoreVec;
    // calculate PAF value of each inner point
    float subScore = 0.0;
    for (int i = 0; i < xs.size(); i++) {
        subScore = pafX.at<float>(ys[i], xs[i]) * vx + pafY.at<float>(ys[i], xs[i]) * vy;
        subScoreVec.push_back(subScore);
    }
    // remove inner points such that has PAF value < K_LOCAL_PAF_SCORE_THRESHOLD
    subScoreVec.erase(std::remove_if(
        subScoreVec.begin(), subScoreVec.end(),
        [](const float &x) {
            return x <= K_LOCAL_PAF_SCORE_THRESHOLD;
        }), subScoreVec.end());
    std::vector<float> result {0.0, 0.0};
    score = std::accumulate(subScoreVec.begin(), subScoreVec.end(), 0.0);
    count = subScoreVec.size();
    result[0] = score / (count + 1e-6);
    result[1] = count;
    return result;
}

/**
 * @brief Remove conflict skeletons
 * @param src - Source vector that stores skeletons to be processed
 * @param dst - Target vector that collects candidate skeletons
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::ConntectionNms(std::vector<PartPair> &src, std::vector<PartPair> &dst)
{
    // Remove conflict skeletons, if two skeletons of the same type share a same end point, they are conflict
    std::vector<int> usedIndex1 {};
    std::vector<int> usedIndex2 {};
    // Sort skeletons in ascending order of affinity score
    std::sort(src.begin(), src.end(), GreaterSort);
    for (int i = 0; i < src.size(); i++) {
        PartPair candidate = src[i];
        if (std::find(usedIndex1.begin(), usedIndex1.end(), candidate.idx1) != usedIndex1.end()
            || std::find(usedIndex2.begin(), usedIndex2.end(), candidate.idx2) != usedIndex2.end()) {
            continue;
        }
        dst.push_back(candidate);
        usedIndex1.push_back(candidate.idx1);
        usedIndex2.push_back(candidate.idx2);
    }
    return APP_ERR_OK;
}

/**
 * @breif Calculate expected confidence of each possible skeleton and choose candidates
 * @param partIndex - Index of skeleton in K_POSE_BODY_PART_SKELETONS
 * @param coor - Candidate positions of endpoints
 * @param coorScore - Corresponding score of coor
 * @param pafHeatmap - PAF heatmap
 * @param connections - Target vector that collects candidate skeletons
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::ScoreSkeletons(const int partIndex,
    const std::vector<std::vector<cv::Point> > &coor,
    const std::vector<std::vector<float> > &coorScore,
    const std::vector<cv::Mat> &pafHeatmap,
    std::vector<PartPair> &connections,
    const std::vector<int> &visionInfos)
{
    // Calculate x y ratios from visionInfo, generally same as the codes in ResizeHeatmaps
    // Calculate padding direction and padding value
    int originHeight = visionInfos[0];
    int originWidth = visionInfos[1];
    // padding along height
    int paddingDirection = 0;
    if (originHeight > originWidth) {
        // padding along width
        paddingDirection = 1;
    }
    int paddingValue = 0;
    // ratios for turnning x,y into original scale
    float x_ratio = (float)originWidth / inputWidth_;
    float y_ratio = (float)originHeight / inputHeight_;
    if (paddingDirection == 0) {
        // pad height
        paddingValue = floor(inputHeight_ - inputWidth_ * originHeight / originWidth);
        y_ratio = (float)originHeight / (float)(inputHeight_ - paddingValue);
    } else {
        // pad width
        paddingValue = floor(inputWidth_ - inputHeight_ * originWidth / originHeight);
        x_ratio = (float)originWidth / (float)(inputWidth_ - paddingValue);
    }
    
    // Use point1 and point2 to represent the two endpoints of a skeleton
    const int indexStride = 2;
    const int endPointNum = 2;
    int cocoSkeletonIndex1 = K_POSE_BODY_PART_SKELETONS[indexStride * partIndex];
    int cocoSkeletonIndex2 = K_POSE_BODY_PART_SKELETONS[indexStride * partIndex + 1];
    int pafXIndex = K_POSE_MAP_INDEX[indexStride * partIndex];
    int pafYIndex = K_POSE_MAP_INDEX[indexStride * partIndex + 1];
    std::vector<cv::Point> endpoints(endPointNum, cv::Point(0, 0));
    std::vector<PartPair> connectionTemp {};
    std::vector<float> result {0.0, 0.0};
    // Calculate the affinity score of each skeleton composed of all candidate point1 and point2
    for (int i = 0; i < coor[cocoSkeletonIndex1].size(); i++) {
        cv::Point point1;
        point1.x = coor[cocoSkeletonIndex1][i].x;
        point1.y = coor[cocoSkeletonIndex1][i].y;
        endpoints[0] = point1;
        for (int j = 0; j < coor[cocoSkeletonIndex2].size(); j++) {
            cv::Point point2;
            point2.x = coor[cocoSkeletonIndex2][j].x;
            point2.y = coor[cocoSkeletonIndex2][j].y;
            endpoints[1] = point2;
            result = OneSkeletonScore(endpoints, pafHeatmap[pafXIndex], pafHeatmap[pafYIndex]);
            // Keep skeletons with affinity scores greater than 0 and
            // valid internal points greater than K_LOCAL_PAF_COUNT_THRESHOLD
            if (result[1] <= K_LOCAL_PAF_COUNT_THRESHOLD || result[0] <= 0.0) {
                continue;
            }
            PartPair skeleton;
            skeleton.score = result[0];
            skeleton.partIdx1 = cocoSkeletonIndex1;
            skeleton.partIdx2 = cocoSkeletonIndex2;
            skeleton.idx1 = i;
            skeleton.idx2 = j;
            // While save the keypoints coordinates, transfer them into original scale
            skeleton.coord1.push_back(point1.x * x_ratio);
            skeleton.coord1.push_back(point1.y * y_ratio);
            skeleton.coord2.push_back(point2.x * x_ratio);
            skeleton.coord2.push_back(point2.y * y_ratio);
            skeleton.score1 = coorScore[cocoSkeletonIndex1][i];
            skeleton.score2 = coorScore[cocoSkeletonIndex2][j];
            connectionTemp.push_back(skeleton);
        }
    }
    // For skeletons with the same endpoints, keep the one with larger affinity score
    ConntectionNms(connectionTemp, connections);
    return APP_ERR_OK;
}

/**
 * @brief Merge a skeleton to an existed person
 * @param personList - Currently existed person list
 * @param currentPair - Skeleton to be merged
 * @return True if merged successfully, otherwise false
 */
bool MxpiRTMOpenposePostProcess::MergeSkeletonToPerson(std::vector<std::vector<PartPair> > &personList,
    PartPair currentPair)
{
    // Use point1 and point2 to represent the two endpoints of a skeleton
    for (int k = 0; k < personList.size(); k++) {
        std::vector<PartPair> &currentPerson = personList[k];
        for (int i = 0; i < currentPerson.size(); i++) {
            if (currentPair.partIdx1 == currentPerson[i].partIdx1 &&
                currentPair.idx1 == currentPerson[i].idx1) {
                // point1 of current skeleton is the same as point1 of a skeleton in current person
                currentPerson.push_back(currentPair);
                return true;
            } else if (currentPair.partIdx1 == currentPerson[i].partIdx2 &&
                currentPair.idx1 == currentPerson[i].idx2) {
                // point1 of current skeleton is the same as point2 of a skeleton in current person
                currentPerson.push_back(currentPair);
                return true;
            } else if (currentPair.partIdx2 == currentPerson[i].partIdx1 &&
                currentPair.idx2 == currentPerson[i].idx1) {
                // point2 of current skeleton is the same as point1 of a skeleton in current person
                currentPerson.push_back(currentPair);
                return true;
            } else if (currentPair.partIdx2 == currentPerson[i].partIdx2 &&
                currentPair.idx2 == currentPerson[i].idx2) {
                // point2 of current skeleton is the same as point2 of a skeleton in current person
                currentPerson.push_back(currentPair);
                return true;
            }
        }
    }
    // Can not merge to any existed person, create new person
    std::vector<PartPair> newPerson {};
    newPerson.push_back(currentPair);
    personList.push_back(newPerson);
    return true;
}

/**
 * @breif Group keypoints to skeletons and assemble them to person
 * @param pafHeatmap - PAF heatmap
 * @param coor - Coordinates of all the candidate keypoints
 * @param coorScore - Corresponding score of coordinates
 * @param personList - Target vector to store person, each person is stored as a vector of skeletons
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::GroupKeypoints(const std::vector<cv::Mat> &pafHeatmap,
    const std::vector<int> &visionInfos,
    const std::vector<std::vector<cv::Point> > &coor,
    const std::vector<std::vector<float> > &coorScore,
    std::vector<std::vector<PartPair> > &personList)
{
    for (int i = 0; i < K_NUM_BODY_PARTS + 1; i++) {
        // Chooose candidate skeletons for each category, there are a total of kNumBodyPart + 1 categories of skeletons
        std::vector<PartPair> partConnections {};
        ScoreSkeletons(i, coor, coorScore, pafHeatmap, partConnections, visionInfos);
        // Merge newly generated skeletons to existed person or create new person
        if (i == 0) {
            // For the first category, each different skeleton of this category stands for different person
            for (int j = 0; j < partConnections.size(); j++) {
                std::vector<PartPair> newPerson {};
                newPerson.push_back(partConnections[j]);
                personList.push_back(newPerson);
            }
        } else if (i == K_NUM_BODY_PARTS - 1 || i == K_NUM_BODY_PARTS) {
            // The last two skeletons do not contribute to person score
            for (int j = 0; j < partConnections.size(); j++) {
                partConnections[j].score = 0;
                partConnections[j].score1 = 0;
                partConnections[j].score2 = 0;
                bool can_merge = MergeSkeletonToPerson(personList, partConnections[j]);
            }
        } else {
            for (int j = 0; j < partConnections.size(); j++) {
                MergeSkeletonToPerson(personList, partConnections[j]);
            }
        }
    }
    return APP_ERR_OK;
}

/**
 * @brief Calculate score of a person according to its skeletons
 * @param person - Target person
 * @return Score value
 */
float MxpiRTMOpenposePostProcess::PersonScore(const std::vector <PartPair> &person)
{
    // The score of a person is composed of the scores of all his keypoints and that of all his skeletons
    std::vector<int> seenKeypoints = {};
    float personScore = 0.0;
    for (int i = 0; i < person.size(); i++) {
        PartPair skeleton = person[i];
        if (std::find(seenKeypoints.begin(), seenKeypoints.end(), skeleton.partIdx1) == seenKeypoints.end()) {
            seenKeypoints.push_back(skeleton.partIdx1);
            personScore += skeleton.score1;
        }
        if (std::find(seenKeypoints.begin(), seenKeypoints.end(), skeleton.partIdx2) == seenKeypoints.end()) {
            seenKeypoints.push_back(skeleton.partIdx2);
            personScore += skeleton.score2;
        }
        personScore += skeleton.score;
    }
    // Ignore person whose number of skeletons is less than K_PERSON_SKELETON_COUNT_THRESHOLD or
    // the average score of each keypoint is less than K_PERSON_KEYPOINT_AVG_SCORE_THRESHOLD
    if (seenKeypoints.size() < K_PERSON_SKELETON_COUNT_THRESHOLD ||
        (personScore / seenKeypoints.size()) < K_PERSON_KEYPOINT_AVG_SCORE_THRESHOLD) {
        return 0.0;
    }
    return personScore;
}

/**
 * @brief Prepare output in the format of MxpiPersonList
 * @param personList - Source data in the format of std::vector<std::vector<PartPair> >
 * @param dstMxpiPersonList - Target data in the format of MxpiPersonList
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::GenerateMxpiOutput(const std::vector<std::vector<PartPair> > &personList,
    MxTools::MxpiOsdInstancesList &dstMxpiOsdInstancesList)
{
    const float floatEqualZeroBias = 0.000001;
    auto mxpiOsdInstancePtr = dstMxpiOsdInstancesList.add_osdinstancesvec();
    MxTools::MxpiMetaHeader *dstOsdMxpiMetaheaderList = mxpiOsdInstancePtr->add_headervec();
    dstOsdMxpiMetaheaderList->set_datasource(parentName_);
    dstOsdMxpiMetaheaderList->set_memberid(0);
    for (int k = 0; k < personList.size(); k++) {
        float personScore = PersonScore(personList[k]);
        // Ignore person with score 0
        if (fabs(personScore - 0) < floatEqualZeroBias) {
            continue;
        }
        int pointResult[18] = {0};
        
        for (int j = 0; j < personList[k].size(); j++) {
            PartPair skeleton = personList[k][j];
            auto mxpiLinePtr = mxpiOsdInstancePtr->add_osdlinevec();
            mxpiLinePtr->set_x0(skeleton.coord1[0]);
            mxpiLinePtr->set_y0(skeleton.coord1[1]);
            mxpiLinePtr->set_x1(skeleton.coord2[0]);
            mxpiLinePtr->set_y1(skeleton.coord2[1]);
            // Create osdparams object by mutable_ method
            MxTools::MxpiOsdParams* inlineparams = mxpiLinePtr->mutable_osdparams();
            inlineparams->set_scalorb(RGB_BLUE_VALUE);
            inlineparams->set_scalorg(RGB_GREEN_VALUE);
            inlineparams->set_scalorr(RGB_RED_VALUE);
            inlineparams->set_thickness(THICKNESS_OF_LINES);
            inlineparams->set_linetype(LINETYPE_OF_LINES);
            inlineparams->set_shift(0);
            // Create osdobjects for keypoints
            if (pointResult[skeleton.partIdx1] == 0) {
                pointResult[skeleton.partIdx1] = 1;
                MxTools::MxpiOsdCircle* dstOsdCircle = mxpiOsdInstancePtr->add_osdcirclevec();
                dstOsdCircle->set_x0(skeleton.coord1[0]);
                dstOsdCircle->set_y0(skeleton.coord1[1]);
                dstOsdCircle->set_radius(RADIU_OF_DOTS);
                MxTools::MxpiOsdParams* incircleparams = dstOsdCircle->mutable_osdparams();
                incircleparams->set_scalorb(RGB_BLUE_VALUE);
                incircleparams->set_scalorg(RGB_GREEN_VALUE);
                incircleparams->set_scalorr(RGB_RED_VALUE);
                incircleparams->set_thickness(THICKNESS_OF_LINES);
                incircleparams->set_linetype(LINETYPE_OF_LINES);
                incircleparams->set_shift(0);
            }
            if (pointResult[skeleton.partIdx2] == 0) {
                pointResult[skeleton.partIdx2] = 1;
                MxTools::MxpiOsdCircle* dstOsdCircle = mxpiOsdInstancePtr->add_osdcirclevec();
                dstOsdCircle->set_x0(skeleton.coord2[0]);
                dstOsdCircle->set_y0(skeleton.coord2[1]);
                dstOsdCircle->set_radius(RADIU_OF_DOTS);
                MxTools::MxpiOsdParams* incircleparams = dstOsdCircle->mutable_osdparams();
                incircleparams->set_scalorb(RGB_BLUE_VALUE);
                incircleparams->set_scalorg(RGB_GREEN_VALUE);
                incircleparams->set_scalorr(RGB_RED_VALUE);
                incircleparams->set_thickness(THICKNESS_OF_LINES);
                incircleparams->set_linetype(LINETYPE_OF_LINES);
                incircleparams->set_shift(0);
            }
        }
    }
    return APP_ERR_OK;
}

/**
 * Overall process to generate all person skeleton information
 * @param imageDecoderVisionListSptr - Source MxpiVisionList containing vision data about input image
 * @param srcMxpiTensorPackage - Source MxpiTensorPackage containing heatmap data
 * @param dstMxpiPersonList - Target MxpiPersonList containing detection result list
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::GeneratePersonList(const MxpiVisionList imageDecoderVisionListSptr,
    const MxpiTensorPackageList srcMxpiTensorPackage,
    MxTools::MxpiOsdInstancesList &dstMxpiOsdInstancesList)
{
    // Get tensor
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(srcMxpiTensorPackage, tensors);
    // Get size of the input image and the aligned image
    std::vector<int> visionInfos = {};
    GetImageSizes(imageDecoderVisionListSptr, visionInfos);
    std::vector<cv::Mat> keypointHeatmap, pafHeatmap;
    // Read data from tensor output by the upstream plugin
    std::vector<std::vector<cv::Mat> > result = ReadDataFromTensorPytorch(tensors);
    keypointHeatmap = result[0];
    pafHeatmap = result[1];
    // Resize heatmaps to the size of the input image
    ResizeHeatmaps(keypointHeatmap, pafHeatmap, visionInfos);
    // Extract candidate keypoints
    std::vector<std::vector<cv::Point> > coor {};
    std::vector<std::vector<float> > coorScore {};
    ExtractKeypoints(keypointHeatmap, coor, coorScore);
    // Group candidate keypoints to candidate skeletons and generate person
    std::vector<std::vector<PartPair> > personList {};
    GroupKeypoints(pafHeatmap, visionInfos, coor, coorScore, personList);
    // Prepare output in the format of MxpiPersonList
    GenerateMxpiOutput(personList, dstMxpiOsdInstancesList);
    return APP_ERR_OK;
}

/**
 * @brief Initialize configure parameter.
 * @param configParamMap
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "MxpiRTMOpenposePostProcess::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    parentName_ = dataSource_;
    std::shared_ptr<string> imageDecoderPropSptr = std::static_pointer_cast<string>(configParamMap["imageSource"]);
    imageDecoderName_ = *imageDecoderPropSptr.get();
    std::shared_ptr<uint32_t > inputHeightPropSptr =
            std::static_pointer_cast<uint32_t >(configParamMap["inputHeight"]);
    inputHeight_ = *inputHeightPropSptr.get();
    std::shared_ptr<uint32_t > inputWidthPropSptr =
            std::static_pointer_cast<uint32_t >(configParamMap["inputWidth"]);
    inputWidth_ = *inputWidthPropSptr.get();
    return APP_ERR_OK;
}

/**
 * @brief DeInitialize configure parameter.
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::DeInit()
{
    LogInfo << "MxpiRTMOpenposePostProcess::DeInit end.";
    LogInfo << "MxpiRTMOpenposePostProcess::DeInit end.";
    return APP_ERR_OK;
}

/**
 * @brief Process the data of MxpiBuffer.
 * @param mxpiBuffer
 * @return APP_ERROR
 */
APP_ERROR MxpiRTMOpenposePostProcess::Process(std::vector<MxpiBuffer*> &mxpiBuffer)
{
    LogInfo << "MxpiRTMOpenposePostProcess::Process start";
    MxpiBuffer *buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) <<
        "MxpiRTMOpenposePostProcess process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiRTMOpenposePostProcess process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the output of tensorinfer from buffer
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    shared_ptr<MxpiTensorPackageList> srcMxpiTensorPackageListSptr
        = static_pointer_cast<MxpiTensorPackageList>(metadata);

    // Get the output of imagedecoder from buffer
    shared_ptr<void> id_metadata = mxpiMetadataManager.GetMetadata(imageDecoderName_);
    shared_ptr<MxpiVisionList> imageDecoderVisionListSptr
            = static_pointer_cast<MxpiVisionList>(id_metadata);

    // Generate output
    shared_ptr<MxTools::MxpiOsdInstancesList> dstMxpiOsdInstancesListSptr =
            make_shared<MxTools::MxpiOsdInstancesList>();
    APP_ERROR ret = GeneratePersonList(*imageDecoderVisionListSptr,
                                       *srcMxpiTensorPackageListSptr, *dstMxpiOsdInstancesListSptr);
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiRTMOpenposePostProcess get skeleton information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiOsdInstancesListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiRTMOpenposePostProcess add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiRTMOpenposePostProcess::Process end";
    return APP_ERR_OK;
}

/**
 * @brief Definition the parameter of configure properties.
 * @return std::vector<std::shared_ptr<void>>
 */
std::vector<std::shared_ptr<void>> MxpiRTMOpenposePostProcess::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto imageDecoderNameProSptr = (std::make_shared<ElementProperty<string>>)(ElementProperty<string> {STRING, "imageSource", "inputName", "the name of imagedecoder", "mxpi_imagedecoder0", "NULL", "NULL"});
    auto inputHeightProSptr = (std::make_shared<ElementProperty<uint32_t>>)(ElementProperty<uint32_t> {UINT, "inputHeight", "inputHeightValue", "the height of the input image", 368, 0, 1000});
    auto inputWidthProSptr = (std::make_shared<ElementProperty<uint32_t>>)(ElementProperty<uint32_t> {UINT, "inputWidth", "inputWidthValue", "the width of the input image", 368, 0, 1000});
    properties.push_back(imageDecoderNameProSptr);
    properties.push_back(inputHeightProSptr);
    properties.push_back(inputWidthProSptr);
    return properties;
}

APP_ERROR MxpiRTMOpenposePostProcess::SetMxpiErrorInfo(MxpiBuffer &buffer, const std::string plugin_name,
    const MxpiErrorInfo mxpiErrorInfo)
{
    APP_ERROR ret = APP_ERR_OK;
    // Define an object of MxpiMetadataManager
    MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(plugin_name, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiRTMOpenposePostProcess)
