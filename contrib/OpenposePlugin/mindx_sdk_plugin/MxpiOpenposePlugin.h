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

#ifndef OPENPOSEPLUGIN_MXPIOPENPOSEPLUGIN_H
#define OPENPOSEPLUGIN_MXPIOPENPOSEPLUGIN_H
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "mxpiOpenposeProto.pb.h"

/**
* @api
* @brief Definition of MxpiOpenposePlugin class.
*/

namespace {
    struct PartPair {
        float score;
        int part_idx1;
        int part_idx2;
        int idx1;
        int idx2;
        std::vector<float> coord1;
        std::vector<float> coord2;
        float score1;
        float score2;
    };
}

namespace MxPlugins {
    class MxpiOpenposePlugin : public MxTools::MxPluginBase {
    public:
        /**
         * @api
         * @brief Initialize configure parameter.
         * @param config_param_map
         * @return APP_ERROR
         */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>>& config_param_map) override;
        /**
         * @api
         * @brief DeInitialize configure parameter.
         * @return APP_ERROR
         */
        APP_ERROR DeInit() override;
        /**
         * @api
         * @brief Process the data of MxpiBuffer.
         * @param mxpi_buffer
         * @return APP_ERROR
         */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer*>& mxpi_buffer) override;
        /**
         * @api
         * @brief Definition the parameter of configure properties.
         * @return std::vector<std::shared_ptr<void>>
         */
        static std::vector<std::shared_ptr<void>> DefineProperties();
        /**
         * @api
         * @brief Overall process to generate all person skeleton information
         * @param src_mxpi_tensor_package - The source MxpiTensorPackage
         * @param dst_mxpi_person_list - The target structure containing detection result list
         * @return APP_ERROR
         */
         APP_ERROR GeneratePersonList(const MxTools::MxpiTensorPackageList src_mxpi_tensor_package,
                                    mxpiopenposeproto::MxpiPersonList& dst_mxpi_person_list);
         /**
          * @api
          * @brief Extract candidate keypoints from output heatmap
          * @param keypoint_heatmap - Keypoint heatmap sotred in vector
          * @param x_coor - Keep xs for candidate keypoints by category
          * @param y_coor - Keep ys for candidate keypoints by category
          * @param channel - Channel number of keypoint heatmap
          * @param height - Height of keypoint heatmap
          * @param width - Width of keypoint heatmap
          * @return APP_ERROR
          */
         APP_ERROR ExtractKeyPoints(std::vector<std::vector<std::vector<float> > > &keypoint_heatmap,
                             std::vector<std::vector<int> > &x_coor,
                             std::vector<std::vector<int> > &y_coor,
                             int channel, int height, int width);
         /**
          * @api
          * @brief Overall process to generate candidate skeletons from output heatmap
          * @param paf_heatmap - PAF heatmap sotred in vector
          * @param keypoint_heatmap - Keypoint heatmap sotred in vector
          * @param x_coor
          * @param y_coor
          * @param connections - Keep skeletons in a vector of PartPair, each element represents a candidate skeleton
          * @return APP_ERROR
          */
         APP_ERROR GenerateSkeletons(const std::vector<std::vector<std::vector<float> > > &paf_heatmap,
                                    const std::vector<std::vector<std::vector<float> > > &keypoint_heatmap,
                                    const std::vector<std::vector<int> > &x_coor,
                                    const std::vector<std::vector<int> > &y_coor,
                                    std::vector<PartPair> &connnections);
         /**
          * @api
          * @brief Generate candidate skeletons for one category by scoring different combinations of endpoints
          * @param part_idx1 - Index of one endpoint of a skeleton
          * @param part_idx2 - Index of the other endpoint of a skeleton
          * @param part1_x - All candidate xs of category part_idx1
          * @param part1_y - All candidate ys of category part_idx1
          * @param part2_x - All candidate xs of category part_idx2
          * @param part2_y - All candidate ys of category part_idx2
          * @param paf_x - PAFX for a skeleton
          * @param paf_y - PAFY for a skeleton
          * @param rescale_height - Used to compute original coordinates
          * @param rescale_width - Used to compute original coordinates
          * @param connections
          * @return APP_ERROR
          */
         APP_ERROR ScoreSkeletons(int part_idx1, int part_idx2,
                                const std::vector<int> &part1_x, const std::vector<int> &part1_y,
                                const std::vector<int> &part2_x, const std::vector<int> &part2_y,
                                const std::vector<std::vector<float> > &paf_x, const std::vector<std::vector<float> > &paf_y,
                                const std::vector<std::vector<std::vector<float> > > &keypoint_heatmap,
                                float rescale_height, float rescale_width,
                                std::vector<PartPair> &connections);
         /**
          * @api
          * @brief Compute expected confidence for each candidate skeleton
          * @param x1 - Coordinate x for one endpoint of this skeleton
          * @param y1 - Coordinate y for one endpoint of this skeleton
          * @param x2 - Coordinate x for the other endpoint of this skeleton
          * @param y2 - Coordinate y for the other endpoint of this skeleton
          * @param paf_x
          * @param paf_y
          * @param result - Keep expected confidence of this skeleton
          * @return APP_ERROR
          */
         APP_ERROR OneSkeletonScore(int x1, int y1, int x2, int y2,
                                                       const std::vector<std::vector<float> > &paf_x,
                                                       const std::vector<std::vector<float> > &paf_y,
                                                       std::vector<int> &result);
         /**
          * @api
          * @breif Assemble multi person from candidate skeletons
          * @param connections - Candidate skeletons
          * @param dst_mxpi_person_list - The target structure containing detected person list
          * @return APP_ERROR
          */
         APP_ERROR AssemblePerson(std::vector <PartPair> &connections,
                                mxpiopenposeproto::MxpiPersonList& dst_mxpi_person_list);
         /**
          * @api
          * @brief Verify if a skeleton belongs to a person
          * @param current_person - Person including multi skeletons
          * @param current_pair - The skeleton
          * @param result - Verification result, true if this skeleton belongs to current_person, false otherwise
          * @return APP_ERROR
          */
         APP_ERROR MergeParts(std::vector<PartPair> &current_person, PartPair current_pair, std::shared_ptr<bool> &result);
        /**
         * @api
         * @brief Find start point to generate a new person, that is the first skeleton that is not merged to any person
         * @param flags - Flag vector marking whether each skeleton has been merged to a person
         * @param next_idx - Keep the index in flags of the first skeleton that is not merged to any person
         * @return APP_ERROR
         */
         APP_ERROR NextPersonStart(std::vector<int> &flags, std::shared_ptr<int> &next_idx);

            private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer& buffer, const std::string plugin_name,
                                   const MxTools::MxpiErrorInfo mxpi_error_info);
        std::string parentName_;
        std::string descriptionMessage_;
        std::ostringstream ErrorInfo_;
    };
}
#endif //OPENPOSEPLUGIN_MXPIOPENPOSEPLUGIN_H
