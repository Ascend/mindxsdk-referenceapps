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
#include "MxpiOpenposePostProcess.h"
#include <numeric>
#include <algorithm>
#include "opencv2/opencv.hpp"
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
}

/**
 * @brief decode MxpiTensorPackageList
 * @param tensorPackageList - Source tensorPackageList
 * @param tensors - Target TensorBase data
 */
static void GetTensors(const MxTools::MxpiTensorPackageList tensor_package_list,
                       std::vector<MxBase::TensorBase> &tensors)
{
    for (int i = 0; i < tensor_package_list.tensorpackagevec_size(); ++i) {
        for (int j = 0; j < tensor_package_list.tensorpackagevec(i).tensorvec_size(); j++) {
            MxBase::MemoryData memory_data = {};
            memory_data.deviceId = tensor_package_list.tensorpackagevec(i).tensorvec(j).deviceid();
            memory_data.type = (MxBase::MemoryData::MemoryType)tensor_package_list.
                    tensorpackagevec(i).tensorvec(j).memtype();
            memory_data.size = (uint32_t)tensor_package_list.
                    tensorpackagevec(i).tensorvec(j).tensordatasize();
            memory_data.ptrData = (void *)tensor_package_list.
                    tensorpackagevec(i).tensorvec(j).tensordataptr();
            std::vector<uint32_t> output_shape = {};
            for (int k = 0; k < tensor_package_list.
                    tensorpackagevec(i).tensorvec(j).tensorshape_size(); ++k) {
                output_shape.push_back((uint32_t)tensor_package_list.
                        tensorpackagevec(i).tensorvec(j).tensorshape(k));
            }
            MxBase::TensorBase tmp_tensor(memory_data, true, output_shape,
                                         (MxBase::TensorDataType)tensor_package_list.
                                                 tensorpackagevec(i).tensorvec(j).tensordatatype());
            tensors.push_back(tmp_tensor);
        }
    }
}

/**
 * @breif Get infomation of origin input image and aligned image
 * @param vision_list - MxpiVisionList object obtained from upstream plugin
 * @param vision_infos - Result vector with elements [origin_height, origin_width, aligned_height, aligned_width]
 */
static void GetImageSizes(const MxTools::MxpiVisionList vision_list, std::vector<int> &vision_infos)
{
    MxpiVision vision = vision_list.visionvec(0);
    MxpiVisionInfo vision_info = vision.visioninfo();
    vision_infos.push_back(vision_info.height());
    vision_infos.push_back(vision_info.width());
    vision_infos.push_back(vision_info.heightaligned());
    vision_infos.push_back(vision_info.widthaligned());
}

/**
 * @brief Parsing TensorBase data to keypoint heatmap and PAF heatmap of openpose model
 * @param tensors - TensorBase vector
 * @return Two-element vector, keeping keypoint heatmap and paf heatmap respectively
 */
static std::vector<std::vector<cv::Mat> > ReadDataFromTensorPytorch(const std::vector <MxBase::TensorBase> &tensors)
{
    auto shape = tensors[0].GetShape();
    int channel_keypoint = shape[1];
    int height_index = 2, width_index = 3;
    int height = shape[height_index];
    int width = shape[width_index];
    auto shape_p = tensors[1].GetShape();
    int channel_paf = shape_p[1];
    // Read keypoint data
    auto dataPtr = (uint8_t *)tensors[0].GetBuffer();
    std::shared_ptr<void> keypoint_pointer;
    keypoint_pointer.reset(dataPtr, g_uint8Deleter);
    std::vector<cv::Mat> keypoint_heatmap {};
    int idx = 0;
    for (int i = 0; i < channel_keypoint; i++) {
        cv::Mat single_channel_mat(height, width, CV_32FC1, cv::Scalar(0));
        for (int j = 0; j < height; j++) {
            float *ptr = single_channel_mat.ptr<float>(j);
            for (int k = 0; k < width;  k++) {
                ptr[k] = static_cast<float *>(keypoint_pointer.get())[idx];
                idx += 1;
            }
        }
        keypoint_heatmap.push_back(single_channel_mat);
    }
    // Read PAF data
    auto data_paf_ptr = (uint8_t *)tensors[1].GetBuffer();
    std::shared_ptr<void> paf_pointer;
    paf_pointer.reset(data_paf_ptr, g_uint8Deleter);
    std::vector<cv::Mat> paf_heatmap {};
    idx = 0;
    for (int i = 0; i < channel_paf; i++) {
        cv::Mat single_channel_mat(height, width, CV_32FC1, cv::Scalar(0));
        for (int j = 0; j < height; j++) {
            float *ptr = single_channel_mat.ptr<float>(j);
            for (int k = 0; k < width;  k++) {
                ptr[k] = static_cast<float *>(paf_pointer.get())[idx];
                idx += 1;
            }
        }
        paf_heatmap.push_back(single_channel_mat);
    }
    std::vector<std::vector<cv::Mat> > result = {keypoint_heatmap, paf_heatmap};
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
    cv::Mat plain_with_border;
    const int borderPadding = 2;
    const int bottomRightIndex = 2;
    cv::copyMakeBorder(plain, plain_with_border, borderPadding, borderPadding, borderPadding, borderPadding,
                       BORDER_CONSTANT, cv::Scalar(0));
    cv::Mat plain_with_border_clone = plain_with_border.clone();
    int subMatCols = plain_with_border.cols - borderPadding;
    int subMatRows = plain_with_border.rows - borderPadding;
    cv::Mat plain_center = plain_with_border(cv::Rect(1, 1, subMatCols, subMatRows));
    cv::Mat plain_bottom = plain_with_border(cv::Rect(1, bottomRightIndex, subMatCols, subMatRows));
    cv::Mat plain_up = plain_with_border(cv::Rect(1, 0, subMatCols, subMatRows));
    cv::Mat plain_left = plain_with_border(cv::Rect(0, 1, subMatCols, subMatRows));
    cv::Mat plain_right = plain_with_border(cv::Rect(bottomRightIndex, 1, subMatCols, subMatRows));
    int count = 0;
    for (int i = 0; i < plain_center.rows; i++) {
        float *center_ptr = plain_center.ptr<float>(i);
        float *bottom_ptr = plain_bottom.ptr<float>(i);
        float *up_ptr = plain_up.ptr<float>(i);
        float *left_ptr = plain_left.ptr<float>(i);
        float *right_ptr = plain_right.ptr<float>(i);
        float *clone_border_ptr = plain_with_border_clone.ptr<float>(i + 1);
        for (int j = 0; j < plain_center.cols; j++) {
            if (!((center_ptr[j] > up_ptr[j]) && (center_ptr[j] > bottom_ptr[j]) &&
                (center_ptr[j] > left_ptr[j]) && (center_ptr[j] > right_ptr[j]))) {
                clone_border_ptr[j + 1] = 0;
            }
        }
    }
    plain = plain_with_border_clone(cv::Rect(borderPadding, borderPadding, plain_center.cols - borderPadding,
                                             plain_center.rows - borderPadding)).clone();
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
 * @param keypoint_heatmap - Keypoint heatmap, each channel of the heatmap is stored as a Mat
 * @param paf_heatmap - PAF heatmap, each channel of the heatmap is stored as a Mat
 * @param vision_infos - Vision infos of origin image and aligned image
 * @return APP_ERROR
 */
APP_ERROR MxpiOpenposePostProcess::ResizeHeatmaps(std::vector<cv::Mat> &keypoint_heatmap,
                                                  std::vector<cv::Mat > &paf_heatmap,
                                                  std::vector<int> &vision_infos)
{
    // Calculate padding direction and padding value
    int origin_height = vision_infos[0];
    int origin_width = vision_infos[1];
    // padding along height
    int padding_direction = 0;
    if (origin_height > origin_width) {
        // padding along width
        padding_direction = 1;
    }
    int padding_value = 0;
    if (padding_direction == 0) {
        // pad height
        padding_value = floor(inputHeight_ - inputWidth_ * origin_height / origin_width);
    } else {
        // pad width
        padding_value = floor(inputWidth_ - inputHeight_ * origin_width / origin_height);
    }
    // Channel Split Resize
    int height = keypoint_heatmap[0].rows;
    int width = keypoint_heatmap[0].cols;
    for (int i = 0; i < keypoint_heatmap.size(); i++) {
        cv::Mat single_channel_mat = keypoint_heatmap[i];
        cv::resize(single_channel_mat, single_channel_mat, Size(0, 0),
                   K_UPSAMPLED_STRIDE, K_UPSAMPLED_STRIDE, INTER_CUBIC);
        if (padding_direction == 0) {
            // remove height padding
            single_channel_mat =
                    single_channel_mat(cv::Rect(0, 0,
                                                single_channel_mat.cols, single_channel_mat.rows - padding_value));
        } else {
            // remove width padding
            single_channel_mat =
                    single_channel_mat(cv::Rect(0, 0,
                                                single_channel_mat.cols - padding_value, single_channel_mat.rows));
        }
        cv::resize(single_channel_mat, single_channel_mat, Size(origin_width, origin_height), 0, 0, INTER_CUBIC);
        keypoint_heatmap[i] = single_channel_mat;
    }
    for (int i = 0; i < paf_heatmap.size(); i++) {
        cv::Mat single_channel_mat = paf_heatmap[i];
        cv::resize(single_channel_mat, single_channel_mat, Size(0, 0),
                   K_UPSAMPLED_STRIDE, K_UPSAMPLED_STRIDE, INTER_CUBIC);
        if (padding_direction == 0) {
            single_channel_mat =
                    single_channel_mat(cv::Rect(0, 0,
                                                single_channel_mat.cols, single_channel_mat.rows - padding_value));
        } else {
            single_channel_mat =
                    single_channel_mat(cv::Rect(0, 0,
                                                single_channel_mat.cols - padding_value, single_channel_mat.rows));
        }
        cv::resize(single_channel_mat, single_channel_mat, Size(origin_width, origin_height), 0, 0, INTER_CUBIC);
        paf_heatmap[i] = single_channel_mat;
    }
    return APP_ERR_OK;
}

/**
 * @brief Extract candidate keypoints
 * @param keypoint_heatmap - Resized keypoint heatmap
 * @param coor - Keep extracted result, store a point in a cv::Point object,
 * store keypoints of different channel in different vectors
 * @param coor_score - Scores corresponding to extracted keypoints
 * @return APP_ERROR
 */
APP_ERROR MxpiOpenposePostProcess::ExtractKeypoints(std::vector<cv::Mat> &keypoint_heatmap,
                                                    std::vector<std::vector<cv::Point> > &coor,
                                                    std::vector<std::vector<float> > &coor_score)
{
    int polynomial_exponent = 2;
    for (int i = 0; i < keypoint_heatmap.size() - 1; i++) {
        // NMS
        NMS(keypoint_heatmap[i], K_NMS_THRESHOLD);
        std::vector<cv::Point> non_zero_coordinates;
        cv::findNonZero(keypoint_heatmap[i], non_zero_coordinates);
        std::sort(non_zero_coordinates.begin(), non_zero_coordinates.end(), PointSort);
        std::vector<int> suppressed(non_zero_coordinates.size(), 0);
        std::vector<cv::Point> keypoints_without_nearest {};
        std::vector<float> keypoints_score {};
        // Remove other keypoints within a certain range around one keypoints
        for (int j = 0; j < non_zero_coordinates.size(); j++) {
            if (suppressed[j]) {
                continue;
            }
            int thrown_index = j + 1;
            auto it = std::find_if(std::begin(non_zero_coordinates) + j + 1, std::end(non_zero_coordinates),
                                   [non_zero_coordinates, j, polynomial_exponent](cv::Point p){
                float distance = powf((non_zero_coordinates[j].x - p.x), polynomial_exponent) +
                        powf((non_zero_coordinates[j].y - p.y), polynomial_exponent);
                return sqrtf(distance) < K_NEAREST_KEYPOINTS_THRESHOLD;
            });
            while (it != std::end(non_zero_coordinates)) {
                thrown_index = std::distance(std::begin(non_zero_coordinates) + thrown_index, it) + thrown_index;
                suppressed[thrown_index] = 1;
                it = std::find_if(std::next(it), std::end(non_zero_coordinates),
                                  [non_zero_coordinates, j, polynomial_exponent](cv::Point p){
                    float distance = powf((non_zero_coordinates[j].x - p.x), polynomial_exponent) +
                            powf((non_zero_coordinates[j].y - p.y), polynomial_exponent);
                    return sqrtf(distance) < K_NEAREST_KEYPOINTS_THRESHOLD;
                });
            }
            keypoints_without_nearest.push_back(non_zero_coordinates[j]);
            keypoints_score.push_back(keypoint_heatmap[i].at<float>
                (non_zero_coordinates[j].y, non_zero_coordinates[j].x));
        }
        coor.push_back(keypoints_without_nearest);
        coor_score.push_back(keypoints_score);
    }
    return APP_ERR_OK;
}

/**
 * @brief Compute expected confidence for each candidate skeleton
 * @param endpoints - Coordinates of the two end points of a skeleton
 * @param paf_x - PAF heatmap of x coordinate
 * @param paf_y - PAF heatmap of y coordinate
 * @return result - Keep confidence information of this skeleton in the form:
 * [confidence score, number of successfully hit sub points]
 */
std::vector<float> MxpiOpenposePostProcess::OneSkeletonScore(std::vector<cv::Point> &endpoints,
                                                             const cv::Mat &paf_x, const cv::Mat &paf_y)
{
    int x1 = endpoints[0].x, y1 = endpoints[0].y;
    int x2 = endpoints[1].x, y2 = endpoints[1].y;
    // affinity score of this skeleton
    float score = 0;
    // count: number of valid inner points on this skeleton
    int count = 0, num_inter = 10;
    float dx = x2 - x1;
    float dy = y2 - y1;
    float norm_vec = sqrt(dx * dx + dy * dy);
    float vx = dx / (norm_vec + 1e-6);
    float vy = dy / (norm_vec + 1e-6);
    // generate 10 points equally spaced on this skeleton
    std::vector<int> xs {};
    float step_x = dx / (num_inter - 1);
    float temp_x = 0;
    for (int k = 0; k < num_inter; k++) {
        temp_x = x1 + k * step_x;
        xs.push_back(round(temp_x));
    }
    std::vector<int> ys {};
    float step_y = dy / (num_inter - 1);
    float temp_y = 0;
    for (int k = 0; k < num_inter; k++) {
        temp_y = y1 + k * step_y;
        ys.push_back(round(temp_y));
    }
    std::vector<float> sub_score_vec;
    // calculate PAF value of each inner point
    float sub_score = 0.0;
    for (int i = 0; i < xs.size(); i++) {
        sub_score = paf_x.at<float>(ys[i], xs[i]) * vx + paf_y.at<float>(ys[i], xs[i]) * vy;
        sub_score_vec.push_back(sub_score);
    }
    // remove inner points such that has PAF value < K_LOCAL_PAF_SCORE_THRESHOLD
    sub_score_vec.erase(std::remove_if(
        sub_score_vec.begin(), sub_score_vec.end(),
        [](const float &x)
        {
            return x <= K_LOCAL_PAF_SCORE_THRESHOLD;
        }), sub_score_vec.end());
    std::vector<float> result {0.0, 0.0};
    score = std::accumulate(sub_score_vec.begin(), sub_score_vec.end(), 0.0);
    count = sub_score_vec.size();
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
APP_ERROR MxpiOpenposePostProcess::ConntectionNms(std::vector<PartPair> &src, std::vector<PartPair> &dst)
{
    // Remove conflict skeletons, if two skeletons of the same type share a same end point, they are conflict
    std::vector<int> used_idx1 {};
    std::vector<int> used_idx2 {};
    // Sort skeletons in ascending order of affinity score
    std::sort(src.begin(), src.end(), GreaterSort);
    for (int i = 0; i < src.size(); i++) {
        PartPair candidate = src[i];
        if (std::find(used_idx1.begin(), used_idx1.end(), candidate.idx1) != used_idx1.end()
            || std::find(used_idx2.begin(), used_idx2.end(), candidate.idx2) != used_idx2.end()) {
            continue;
        }
        dst.push_back(candidate);
        used_idx1.push_back(candidate.idx1);
        used_idx2.push_back(candidate.idx2);
    }
    return APP_ERR_OK;
}

/**
 * @breif Calculate expected confidence of each possible skeleton and choose candidates
 * @param part_idx - Index of skeleton in K_POSE_BODY_PART_SKELETONS
 * @param coor - Candidate positions of endpoints
 * @param coor_score - Corresponding score of coor
 * @param paf_heatmap - PAF heatmap
 * @param connections - Target vector that collects candidate skeletons
 * @return APP_ERROR
 */
APP_ERROR MxpiOpenposePostProcess::ScoreSkeletons(const int part_idx,
                                                  const std::vector<std::vector<cv::Point> > &coor,
                                                  const std::vector<std::vector<float> > &coor_score,
                                                  const std::vector<cv::Mat> &paf_heatmap,
                                                  std::vector<PartPair> &connections)
{
    // Use point1 and point2 to represent the two endpoints of a skeleton
    int coco_skeleton_idx1 = K_POSE_BODY_PART_SKELETONS[2 * part_idx];
    int coco_skeleton_idx2 = K_POSE_BODY_PART_SKELETONS[2 * part_idx + 1];
    int index_stride = 2;
    int end_point_num = 2;
    int paf_x_idx = K_POSE_MAP_INDEX[index_stride * part_idx];
    int paf_y_idx = K_POSE_MAP_INDEX[index_stride * part_idx + 1];
    std::vector<cv::Point> endpoints(end_point_num, cv::Point(0, 0));
    std::vector<PartPair> connection_temp {};
    std::vector<float> result {0.0, 0.0};
    // Calculate the affinity score of each skeleton composed of all candidate point1 and point2
    for (int i = 0; i < coor[coco_skeleton_idx1].size(); i++) {
        cv::Point point1;
        point1.x = coor[coco_skeleton_idx1][i].x;
        point1.y = coor[coco_skeleton_idx1][i].y;
        endpoints[0] = point1;
        for (int j = 0; j < coor[coco_skeleton_idx2].size(); j++) {
            cv::Point point2;
            point2.x = coor[coco_skeleton_idx2][j].x;
            point2.y = coor[coco_skeleton_idx2][j].y;
            endpoints[1] = point2;
            result = OneSkeletonScore(endpoints, paf_heatmap[paf_x_idx], paf_heatmap[paf_y_idx]);
            // Keep skeletons with affinity scores greater than 0 and
            // valid internal points greater than K_LOCAL_PAF_COUNT_THRESHOLD
            if (result[1] <= K_LOCAL_PAF_COUNT_THRESHOLD || result[0] <= 0.0) {
                continue;
            }
            // Store the information of a skeleton in a custom structure PartPair
            PartPair skeleton;
            skeleton.score = result[0];
            skeleton.partIdx1 = coco_skeleton_idx1;
            skeleton.partIdx2 = coco_skeleton_idx2;
            skeleton.idx1 = i;
            skeleton.idx2 = j;
            skeleton.coord1.push_back(point1.x);
            skeleton.coord1.push_back(point1.y);
            skeleton.coord2.push_back(point2.x);
            skeleton.coord2.push_back(point2.y);
            skeleton.score1 = coor_score[coco_skeleton_idx1][i];
            skeleton.score2 = coor_score[coco_skeleton_idx2][j];
            connection_temp.push_back(skeleton);
        }
    }
    // For skeletons with the same endpoints, keep the one with larger affinity score
    ConntectionNms(connection_temp, connections);
    return APP_ERR_OK;
}

/**
 * @brief Merge a skeleton to an existed person
 * @param person_list - Currently existed person list
 * @param currentPair - Skeleton to be merged
 * @return True if merged successfully, otherwise false
 */
bool MxpiOpenposePostProcess::MergeSkeletonToPerson(std::vector<std::vector<PartPair> > &person_list,
                                                    PartPair currentPair)
{
    // Use point1 and point2 to represent the two endpoints of a skeleton
    for (int k = 0; k < person_list.size(); k++) {
        std::vector<PartPair> &current_person = person_list[k];
        for (int i = 0; i < current_person.size(); i++) {
            if (currentPair.partIdx1 == current_person[i].partIdx1 &&
                currentPair.idx1 == current_person[i].idx1) {
                // point1 of current skeleton is the same as point1 of a skeleton in current person
                current_person.push_back(currentPair);
                return true;
            } else if (currentPair.partIdx1 == current_person[i].partIdx2 &&
                currentPair.idx1 == current_person[i].idx2) {
                // point1 of current skeleton is the same as point2 of a skeleton in current person
                current_person.push_back(currentPair);
                return true;
            } else if (currentPair.partIdx2 == current_person[i].partIdx1 &&
                currentPair.idx2 == current_person[i].idx1) {
                // point2 of current skeleton is the same as point1 of a skeleton in current person
                current_person.push_back(currentPair);
                return true;
            } else if (currentPair.partIdx2 == current_person[i].partIdx2 &&
                currentPair.idx2 == current_person[i].idx2) {
                // point2 of current skeleton is the same as point2 of a skeleton in current person
                current_person.push_back(currentPair);
                return true;
            }
        }
    }
    // Can not merge to any existed person, create new person
    std::vector<PartPair> new_person {};
    new_person.push_back(currentPair);
    person_list.push_back(new_person);
    return true;
}

/**
 * @breif Group keypoints to skeletons and assemble them to person
 * @param paf_heatmap - PAF heatmap
 * @param coor - Coordinates of all the candidate keypoints
 * @param coor_score - Corresponding score of coordinates
 * @param person_list - Target vector to store person, each person is stored as a vector of skeletons
 * @return APP_ERROR
 */
APP_ERROR MxpiOpenposePostProcess::GroupKeypoints(const std::vector<cv::Mat> &paf_heatmap,
                                                  const std::vector<std::vector<cv::Point> > &coor,
                                                  const std::vector<std::vector<float> > &coor_score,
                                                  std::vector<std::vector<PartPair> > &person_list)
{
    for (int i = 0; i < K_NUM_BODY_PARTS + 1; i++) {
        // Chooose candidate skeletons for each category, there are a total of kNumBodyPart + 1 categories of skeletons
        std::vector<PartPair> part_connections {};
        ScoreSkeletons(i, coor, coor_score, paf_heatmap, part_connections);
        // Merge newly generated skeletons to existed person or create new person
        if (i == 0) {
            // For the first category, each different skeleton of this category stands for different person
            for (int j = 0; j < part_connections.size(); j++) {
                std::vector<PartPair> new_person {};
                new_person.push_back(part_connections[j]);
                person_list.push_back(new_person);
            }
        } else if (i == K_NUM_BODY_PARTS - 1 || i == K_NUM_BODY_PARTS) {
            // The last two skeletons do not contribute to person score
            for (int j = 0; j < part_connections.size(); j++) {
                part_connections[j].score = 0;
                part_connections[j].score1 = 0;
                part_connections[j].score2 = 0;
                bool can_merge = MergeSkeletonToPerson(person_list, part_connections[j]);
            }
        } else {
            for (int j = 0; j < part_connections.size(); j++) {
                MergeSkeletonToPerson(person_list, part_connections[j]);
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
float MxpiOpenposePostProcess::PersonScore(const std::vector <PartPair> &person)
{
    // The score of a person is composed of the scores of all his keypoints and that of all his skeletons
    std::vector<int> seen_keypoints = {};
    float person_score = 0.0;
    for (int i = 0; i < person.size(); i++) {
        PartPair skeleton = person[i];
        if (std::find(seen_keypoints.begin(), seen_keypoints.end(), skeleton.partIdx1) == seen_keypoints.end()) {
            seen_keypoints.push_back(skeleton.partIdx1);
            person_score += skeleton.score1;
        }
        if (std::find(seen_keypoints.begin(), seen_keypoints.end(), skeleton.partIdx2) == seen_keypoints.end()) {
            seen_keypoints.push_back(skeleton.partIdx2);
            person_score += skeleton.score2;
        }
        person_score += skeleton.score;
    }
    // Ignore person whose number of skeletons is less than K_PERSON_SKELETON_COUNT_THRESHOLD or
    // the average score of each keypoint is less than K_PERSON_KEYPOINT_AVG_SCORE_THRESHOLD
    if (seen_keypoints.size() < K_PERSON_SKELETON_COUNT_THRESHOLD ||
        (person_score / seen_keypoints.size()) < K_PERSON_KEYPOINT_AVG_SCORE_THRESHOLD) {
        return 0.0;
    }
    return person_score;
}

/**
 * @brief Prepare output in the format of MxpiPersonList
 * @param person_list - Source data in the format of std::vector<std::vector<PartPair> >
 * @param dst_mxpi_person_list - Target data in the format of MxpiPersonList
 * @return APP_ERROR
 */
APP_ERROR MxpiOpenposePostProcess::GenerateMxpiOutput(const std::vector<std::vector<PartPair> > &person_list,
                                                      mxpiopenposeproto::MxpiPersonList &dst_mxpi_person_list)
{
    float float_equal_zero_bias = 0.000001;
    for (int k = 0; k < person_list.size(); k++) {
        float person_score = PersonScore(person_list[k]);
        // Ignore person with score 0
        if (fabs(person_score - 0) < float_equal_zero_bias) {
            continue;
        }
        auto mxpi_person_ptr = dst_mxpi_person_list.add_personinfovec();
        mxpiopenposeproto::MxpiMetaHeader *dst_person_mxpi_metaheader_list = mxpi_person_ptr->add_headervec();
        dst_person_mxpi_metaheader_list->set_datasource(parentName_);
        dst_person_mxpi_metaheader_list->set_memberid(0);
        mxpi_person_ptr->set_score(person_score);
        for (int j = 0; j < person_list[k].size(); j++) {
            PartPair skeleton = person_list[k][j];
            auto mxpi_skeleton_ptr = mxpi_person_ptr->add_skeletoninfovec();
            mxpi_skeleton_ptr->set_cocoskeletonindex1(skeleton.partIdx1);
            mxpi_skeleton_ptr->set_cocoskeletonindex2(skeleton.partIdx2);
            mxpi_skeleton_ptr->set_x0(skeleton.coord1[0]);
            mxpi_skeleton_ptr->set_y0(skeleton.coord1[1]);
            mxpi_skeleton_ptr->set_point1score(skeleton.score1);
            mxpi_skeleton_ptr->set_x1(skeleton.coord2[0]);
            mxpi_skeleton_ptr->set_y1(skeleton.coord2[1]);
            mxpi_skeleton_ptr->set_point2score(skeleton.score2);
            mxpi_skeleton_ptr->set_skeletonscore(skeleton.score);
        }
    }
    return APP_ERR_OK;
}

/**
 * Overall process to generate all person skeleton information
 * @param image_decoder_visionListSptr - Source MxpiVisionList containing vision data about input image
 * @param src_mxpi_tensor_package - Source MxpiTensorPackage containing heatmap data
 * @param dst_mxpi_person_list - Target MxpiPersonList containing detection result list
 * @return APP_ERROR
 */
APP_ERROR MxpiOpenposePostProcess::GeneratePersonList(const MxpiVisionList image_decoder_visionListSptr,
                                                      const MxpiTensorPackageList src_mxpi_tensor_package,
                                                      mxpiopenposeproto::MxpiPersonList &dst_mxpi_person_list)
{
    // Get tensor
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(src_mxpi_tensor_package, tensors);
    // Get size of the input image and the aligned image
    std::vector<int> vision_infos = {};
    GetImageSizes(image_decoder_visionListSptr, vision_infos);
    std::vector<cv::Mat> keypoint_heatmap, paf_heatmap;
    // Read data from tensor output by the upstream plugin
    std::vector<std::vector<cv::Mat> > result = ReadDataFromTensorPytorch(tensors);
    keypoint_heatmap = result[0];
    paf_heatmap = result[1];
    // Resize heatmaps to the size of the input image
    ResizeHeatmaps(keypoint_heatmap, paf_heatmap, vision_infos);
    // Extract candidate keypoints
    std::vector<std::vector<cv::Point> > coor {};
    std::vector<std::vector<float> > coor_score {};
    ExtractKeypoints(keypoint_heatmap, coor, coor_score);
    // Group candidate keypoints to candidate skeletons and generate person
    std::vector<std::vector<PartPair> > person_list {};
    GroupKeypoints(paf_heatmap, coor, coor_score, person_list);
    // Prepare output in the format of MxpiPersonList
    GenerateMxpiOutput(person_list, dst_mxpi_person_list);
    return APP_ERR_OK;
}

/**
 * @brief Initialize configure parameter.
 * @param config_param_map
 * @return APP_ERROR
 */
APP_ERROR MxpiOpenposePostProcess::Init(std::map<std::string, std::shared_ptr<void>> &config_param_map)
{
    LogInfo << "MxpiOpenposePostProcess::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parent_name_prop_sptr = std::static_pointer_cast<string>(config_param_map["dataSource"]);
    parentName_ = *parent_name_prop_sptr.get();
    std::shared_ptr<string> image_decoder_prop_sptr = std::static_pointer_cast<string>(config_param_map["imageSource"]);
    imageDecoderName_ = *image_decoder_prop_sptr.get();
    std::shared_ptr<uint32_t > input_height_prop_sptr =
            std::static_pointer_cast<uint32_t >(config_param_map["inputHeight"]);
    inputHeight_ = *input_height_prop_sptr.get();
    std::shared_ptr<uint32_t > input_width_prop_sptr =
            std::static_pointer_cast<uint32_t >(config_param_map["inputWidth"]);
    inputWidth_ = *input_width_prop_sptr.get();
    return APP_ERR_OK;
}

/**
 * @brief DeInitialize configure parameter.
 * @return APP_ERROR
 */
APP_ERROR MxpiOpenposePostProcess::DeInit()
{
    LogInfo << "MxpiOpenposePostProcess::DeInit end.";
    LogInfo << "MxpiOpenposePostProcess::DeInit end.";
    return APP_ERR_OK;
}

/**
 * @brief Process the data of MxpiBuffer.
 * @param mxpi_buffer
 * @return APP_ERROR
 */
APP_ERROR MxpiOpenposePostProcess::Process(std::vector<MxpiBuffer*> &mxpi_buffer)
{
    LogInfo << "MxpiOpenposePostProcess::Process start";
    MxpiBuffer *buffer = mxpi_buffer[0];
    MxpiMetadataManager mxpi_metadata_manager(*buffer);
    MxpiErrorInfo mxpi_error_info;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpi_metadata_manager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) <<
        "MxpiOpenposePostProcess process is not implemented";
        mxpi_error_info.ret = APP_ERR_COMM_FAILURE;
        mxpi_error_info.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpi_error_info);
        LogError << "MxpiOpenposePostProcess process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the output of tensorinfer from buffer
    shared_ptr<void> metadata = mxpi_metadata_manager.GetMetadata(parentName_);
    shared_ptr<MxpiTensorPackageList> src_mxpi_tensor_packageListSptr
	    = static_pointer_cast<MxpiTensorPackageList>(metadata);

    // Get the output of imagedecoder from buffer
    shared_ptr<void> id_metadata = mxpi_metadata_manager.GetMetadata(imageDecoderName_);
    shared_ptr<MxpiVisionList> image_decoder_visionListSptr
            = static_pointer_cast<MxpiVisionList>(id_metadata);

    // Generate output
    shared_ptr<mxpiopenposeproto::MxpiPersonList> dst_mxpi_person_listSptr =
            make_shared<mxpiopenposeproto::MxpiPersonList>();
    APP_ERROR ret = GeneratePersonList(*image_decoder_visionListSptr,
                                       *src_mxpi_tensor_packageListSptr, *dst_mxpi_person_listSptr);
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiOpenposePostProcess get skeleton information failed.";
        mxpi_error_info.ret = ret;
        mxpi_error_info.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpi_error_info);
        return ret;
    }
    ret = mxpi_metadata_manager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dst_mxpi_person_listSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiOpenposePostProcess add metadata failed.";
        mxpi_error_info.ret = ret;
        mxpi_error_info.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpi_error_info);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiOpenposePostProcess::Process end";
    return APP_ERR_OK;
}

/**
 * @brief Definition the parameter of configure properties.
 * @return std::vector<std::shared_ptr<void>>
 */
std::vector<std::shared_ptr<void>> MxpiOpenposePostProcess::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parent_name_pro_sptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
            STRING, "dataSource", "parentName", "the name of previous plugin", "mxpi_modelinfer0", "NULL", "NULL"});
    auto image_decoder_name_pro_sptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
            STRING, "imageSource", "inputName", "the name of imagedecoder", "mxpi_imagedecoder0", "NULL", "NULL"});
    auto input_height_pro_sptr = std::make_shared<ElementProperty<uint32_t>>(ElementProperty<uint32_t>{
            UINT, "inputHeight", "inputHeightValue", "the height of the input image", 368, 0, 1000});
    auto input_width_pro_sptr = std::make_shared<ElementProperty<uint32_t>>(ElementProperty<uint32_t>{
            UINT, "inputWidth", "inputWidthValue", "the width of the input image", 368, 0, 1000});
    properties.push_back(image_decoder_name_pro_sptr);
    properties.push_back(parent_name_pro_sptr);
    properties.push_back(input_height_pro_sptr);
    properties.push_back(input_width_pro_sptr);
    return properties;
}

APP_ERROR MxpiOpenposePostProcess::SetMxpiErrorInfo(MxpiBuffer &buffer, const std::string plugin_name,
                                                    const MxpiErrorInfo mxpi_error_info)
{
    APP_ERROR ret = APP_ERR_OK;
    // Define an object of MxpiMetadataManager
    MxpiMetadataManager mxpi_metadata_manager(buffer);
    ret = mxpi_metadata_manager.AddErrorInfo(plugin_name, mxpi_error_info);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiOpenposePostProcess)
