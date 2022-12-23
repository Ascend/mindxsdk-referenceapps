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

#ifndef OPENPOSEPOSTPROCESS_MXPIRTMOPENPOSEPOSTPROCESS_H
#define OPENPOSEPOSTPROCESS_MXPIRTMOPENPOSEPOSTPROCESS_H
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "opencv2/opencv.hpp"
#include "MxpiRTMOpenposeProtoEval.pb.h"


/**
* @api
* @brief Definition of MxpiRTMOpenposePostProcess class.
*/

namespace MxPlugins {
    struct PartPair {
        float score;
        int partIdx1;
        int partIdx2;
        int idx1;
        int idx2;
        std::vector<float> coord1;
        std::vector<float> coord2;
        float score1;
        float score2;
    };

    class MxpiRTMOpenposePostProcess : public MxTools::MxPluginBase {
    public:
        /**
         * @brief Initialize configure parameter.
         * @param configParamMap
         * @return APP_ERROR
         */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) override;

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
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer*> &mxpiBuffer) override;

        /**
         * @brief Definition the parameter of configure properties.
         * @return std::vector<std::shared_ptr<void>>
         */
        static std::vector<std::shared_ptr<void>> DefineProperties();

        /**
         * Overall process to generate all person skeleton information
         * @param imageDecoderVisionListSptr - Source MxpiVisionList containing vision data about input image
         * @param srcMxpiTensorPackage - Source MxpiTensorPackage containing heatmap data
         * @param dstMxpiPersonList - Target MxpiPersonList containing detection result list
         * @return APP_ERROR
         */
        APP_ERROR GeneratePersonList(const MxTools::MxpiVisionList imageDecoderVisionListSptr,
                                     const MxTools::MxpiTensorPackageList srcMxpiTensorPackage,
                                     MxpiRTMOpenposeProtoEval::MxpiPersonList& dstMxpiPersonList);

        /**
         * @brief Resize output heatmaps to the size of the origin image
         * @param keypointHeatmap - Keypoint heatmap, each channel of the heatmap is stored as a Mat
         * @param pafHeatmap - PAF heatmap, each channel of the heatmap is stored as a Mat
         * @param visionInfos - Vision infos of origin image and aligned image
         * @return APP_ERROR
         */
        APP_ERROR ResizeHeatmaps(std::vector<cv::Mat> &keypointHeatmap,
                                 std::vector<cv::Mat > &pafHeatmap,
                                 std::vector<int> &visionInfos);

         /**
          * @brief Extract candidate keypoints from output heatmap
          * @param keypointHeatmap - Resized keypoint heatmap
          * @param coor - Keep extracted result, store a point in a cv::Point object,
          * store keypoints of different channel in different vectors
          * @param coorScore - Scores corresponding to extracted keypoints
          * @return APP_ERROR
          */
        APP_ERROR ExtractKeypoints(std::vector<cv::Mat> &keypointHeatmap,
                                   std::vector<std::vector<cv::Point> > &coor,
                                   std::vector<std::vector<float> > &coorScore);

        /**
         * @breif Group keypoints to skeletons and assemble them to person
         * @param pafHeatmap - PAF heatmap
         * @param coor - Coordinates of all the candidate keypoints
         * @param coorScore - Corresponding score of coordinates
         * @param personList - Target vector to store person, each person is stored as a vector of skeletons
         * @return APP_ERROR
         */
        APP_ERROR GroupKeypoints(const std::vector<cv::Mat>& pafHeatmap,
                                 const std::vector<int> &visionInfos,
                                 const std::vector<std::vector<cv::Point> > &coor,
                                 const std::vector<std::vector<float> > &coorScore,
                                 std::vector<std::vector<PartPair> > &personList);

        /**
         * @breif Calculate expected confidence of each possible skeleton and choose candidates
         * @param partIndex - Index of skeleton in kPoseBodyPartSkeletons
         * @param coor - Candidate positions of endpoints
         * @param coorScore - Corresponding score of coor
         * @param pafHeatmap - PAF heatmap
         * @param connections - Target vector that collects candidate skeletons
         * @return APP_ERROR
         */
        APP_ERROR ScoreSkeletons(const int partIndex,
                                 const std::vector<std::vector<cv::Point> > &coor,
                                 const std::vector<std::vector<float> > &coorScore,
                                 const std::vector<cv::Mat> &pafHeatmap,
                                 std::vector<PartPair> &connections,
                                 const std::vector<int> &visionInfos);

        /**
         * @brief Compute expected confidence for each candidate skeleton
         * @param endpoints - Coordinates of the two end points of a skeleton
         * @param pafX - PAF heatmap of x coordinate
         * @param pafY - PAF heatmap of y coordinate
         * @return result - Keep confidence information of this skeleton in the form:
         * [confidence score, number of successfully hit sub points]
         */
        std::vector<float> OneSkeletonScore(std::vector<cv::Point> &endpoints,
                                            const cv::Mat &pafX, const cv::Mat &pafY);

        /**
         * @brief Remove duplicate skeletons
         * @param src - Source vector that stores skeletons to be processed
         * @param dst - Target vector that collects filter skeletons
         * @return APP_ERROR
         */
        APP_ERROR ConntectionNms(std::vector<PartPair> &src, std::vector<PartPair> &dst);

        /**
         * @brief Merge a skeleton to an existed person
         * @param personList - Currently existed person list
         * @param currentPair - Skeleton to be merged
         * @return True if merged successfully, otherwise false
         */
        bool MergeSkeletonToPerson(std::vector<std::vector<PartPair> > &personList, PartPair currentPair);

        /**
         * @brief Calculate score of a person according to its skeletons
         * @param person - Target person
         * @return Score value
         */
        float PersonScore(const std::vector<PartPair> &person);

        /**
         * @brief Prepare output in the format of MxpiPersonList
         * @param personList - Source data in the format of std::vector<std::vector<PartPair> >
         * @param dstMxpiPersonList - Target data in the format of MxpiPersonList
         * @return
         */
        APP_ERROR GenerateMxpiOutput(const std::vector<std::vector<PartPair> > &personList,
                                     MxpiRTMOpenposeProtoEval::MxpiPersonList &dstMxpiPersonList);

    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer &buffer, const std::string plugin_name,
                                   const MxTools::MxpiErrorInfo mxpiErrorInfo);
        std::string parentName_;
        std::string imageDecoderName_;
        std::uint32_t inputHeight_;
        std::uint32_t inputWidth_;
        std::ostringstream ErrorInfo_;
    };
}
#endif // OPENPOSEPOSTPROCESS_MXPIRTMOPENPOSEPOSTPROCESS_H

