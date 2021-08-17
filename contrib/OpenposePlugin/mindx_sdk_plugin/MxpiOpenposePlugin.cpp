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
#include "MxpiOpenposePlugin.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"

using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string kSampleKey = "MxpiTensorPackageList";
    auto uint8Deleter = [] (uint8_t* p) { };
    const int kNumBodyParts = 18;

    // CocoSkeletonsNetwork
    const std::vector<unsigned int> kPoseMapIndex {
        12,13, 20,21, 14,15, 16,17, 22,23, 24,25, 0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 28,29, 30,31, 34,35, 32,33, 36,37, 18,19, 26,27
    };

    // CocoSkeletons
    const std::vector<unsigned int> kPoseBodyPartSkeletons {
        1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,  2,16,  5,17
    };

    const float kNmsThreshold = 0.15;
    const float kLocalPafThreshold = 0.2;
    const float kPafCountThreshold = 5;
}

/**
 * @brief decode MxpiTensorPackageList
 * @param tensorPackageList - Source tensorPackageList
 * @param tensors - Target TensorBase data
 */
void GetTensors(const MxTools::MxpiTensorPackageList tensorPackageList,
                std::vector<MxBase::TensorBase> &tensors) {
    for (int i = 0; i < tensorPackageList.tensorpackagevec_size(); ++i) {
        for (int j = 0; j < tensorPackageList.tensorpackagevec(i).tensorvec_size(); j++) {
            MxBase::MemoryData memory_data = {};
            memory_data.deviceId = tensorPackageList.tensorpackagevec(i).tensorvec(j).deviceid();
            memory_data.type = (MxBase::MemoryData::MemoryType)tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).memtype();
            memory_data.size = (uint32_t) tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordatasize();
            memory_data.ptrData = (void *) tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordataptr();
            std::vector<uint32_t> output_shape = {};
            for (int k = 0; k < tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensorshape_size(); ++k) {
                output_shape.push_back((uint32_t) tensorPackageList.
                        tensorpackagevec(i).tensorvec(j).tensorshape(k));
            }
            MxBase::TensorBase tmp_tensor(memory_data, true, output_shape,
                                         (MxBase::TensorDataType)tensorPackageList.
                                                 tensorpackagevec(i).tensorvec(j).tensordatatype());
            tensors.push_back(tmp_tensor);
        }
    }
}

/**
 * @brief Parsing TensorBase data to keypoint heatmap and PAF heatmap of openpose model
 * @param tensors - TensorBase vector
 * @param keypoint_heatmap - Keep keypoint data
 * @param paf_heatmap - Keep PAF data
 * @param channel_keypoint - Channel number of keypoint heatmap
 * @param channel_paf - Channel number of PAF heatmap
 * @param height - Height of two heatmaps
 * @param width - Width of two heatmaps
 */
void ReadDataFromTensorCaffe(const std::vector <MxBase::TensorBase> &tensors,
                                               std::vector<std::vector<std::vector<float> > > &keypoint_heatmap,
                                               std::vector<std::vector<std::vector<float> > > &paf_heatmap,
                                               int channel_keypoint, int channel_paf, int height, int width) {
    auto dataPtr = (uint8_t *)tensors[1].GetBuffer();
    std::shared_ptr<void> keypoint_pointer;
    keypoint_pointer.reset(dataPtr, uint8Deleter);
    int idx = 0;
    float temp_data = 0.0;
    for (int i = 0; i < channel_keypoint; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width;  k++) {
                temp_data = static_cast<float *>(keypoint_pointer.get())[idx];
                if (temp_data < 0) {
                    temp_data = 0;
                }
                keypoint_heatmap[i][j][k] = temp_data;
                idx += 1;
            }
        }
    }

    auto data_paf_ptr = (uint8_t *)tensors[0].GetBuffer();
    std::shared_ptr<void> paf_pointer;
    paf_pointer.reset(data_paf_ptr, uint8Deleter);
    idx = 0;
    temp_data = 0.0;
    for (int i = 0; i < channel_paf; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width;  k++) {
                temp_data = static_cast<float *>(paf_pointer.get())[idx];
                paf_heatmap[i][j][k] = temp_data;
                idx += 1;
            }
        }
    }
}

/**
 * @brief Non-Maximum Suppression
 * @param plain - 2D data for NMS
 * @param height - Height of plain
 * @param width - Width of plain
 * @param threshold - NMS threshold
 */
void NMS(std::vector<std::vector<float> > &plain, int height, int width, float threshold) {
    // Keep points with score below the threshold as 0
    std::vector<std::vector<float> > dst_outer = {};
    for (int i = 0; i < height; i++) {
        std::vector<float> dst_inner = {};
        for (int j = 0; j < width; j++) {
            if (plain[i][j] < threshold) {
                plain[i][j] = 0;
            }
            dst_inner.push_back(plain[i][j]);
        }
        dst_outer.push_back(dst_inner);
    }
    // max filter
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width -1; j++) {
            float maxt = 0;
            for (int k = i - 1; k <= i + 1; k++) {
                for (int m = j - 1; m <= j + 1; m++) {
                    if (maxt < plain[k][m])
                        maxt = plain[k][m];
                }
            }
            dst_outer[i][j] = maxt;
        }
    }
    // only keep the local maximum keypoints
    for (int i = 0 ; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (plain[i][j] != dst_outer[i][j])
                plain[i][j] = 0;
        }
    }
}

/**
 * @brief Comparation between two PartPair elements
 * @param p1 - PartPair p1
 * @param p2 - PartPair p2
 * @return True if the score of p1 is greater than that of p2
 */
bool GreaterSort(PartPair p1, PartPair p2) {
    return p1.score > p2.score;
}

APP_ERROR MxpiOpenposePlugin::ExtractKeyPoints(std::vector<std::vector<std::vector<float> > > &keypoint_heatmap,
                                            std::vector<std::vector<int> > &x_coor,
                                            std::vector<std::vector<int> > &y_coor,
                                            int channel, int height, int width) {
    // NMS
    for (int i = 0; i < channel; i++) {
        NMS(keypoint_heatmap[i], height, width, kNmsThreshold);
    }
    // Get Coor
    for (int i = 0; i < channel; i++) {
        std::vector<int> x_vec = {};
        std::vector<int> y_vec = {};
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                if (keypoint_heatmap[i][j][k] >= kNmsThreshold) {
                    y_vec.push_back(j);
                    x_vec.push_back(k);
                }
            }
        }
        x_coor.push_back(x_vec);
        y_coor.push_back(y_vec);
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiOpenposePlugin::OneSkeletonScore(int x1, int y1, int x2, int y2,
                                             const std::vector<std::vector<float> > &paf_x,
                                             const std::vector<std::vector<float> > &paf_y,
                                             std::vector<int> &result) {
    int score = 0;
    int count = 0;
    int __num_inter = 10;
    float __num_inter_f = 10.0;
    int dx = x2 - x1; // < 0
    int dy = y2 - y1;
    float norm_vec = sqrt(dx * dx + dy * dy);
    if (norm_vec < 1e-4) {
        result[0] = score;
        result[1] = count;
        return APP_ERR_OK;
    }
    float vx = dx / norm_vec;
    float vy = dy / norm_vec;
    std::vector<int> xs {};
    float step_x = dx / __num_inter_f;
    float temp_x;
    if (x1 != x2) {
        for (int k = 0; k < __num_inter; k++) {
            temp_x = x1 + k * step_x;
            if ((temp_x > x2) && (x1 < x2) || (temp_x < x2) && (x1 > x2))
                temp_x = x2;
            xs.push_back(temp_x + 0.5);
        }
    }
    else {
        for (int i = 0; i < __num_inter; i++)
            xs.push_back(x1 + 0.5);
    }
    std::vector<int> ys {};
    if (y1 != y2) {
        float step_y = dy / __num_inter_f;
        float temp_y;
        for (int k = 0; k < __num_inter; k++) {
            temp_y = y1 + k * step_y;
            if ((temp_y > y2) && (y1 < y2) || (temp_y < y2) && (y1 > y2))
                temp_y = y2;
            ys.push_back(temp_y + 0.5);
        }
    }
    else {
        for (int i = 0; i < __num_inter; i++)
            ys.push_back(y1 + 0.5);
    }
    std::vector<int> paf_xs(__num_inter, 0);
    std::vector<int> paf_ys(__num_inter, 0);
    std::vector<float> local_scores {};
    float sub_score;
    for (int i = 0; i < xs.size(); i++) {
        sub_score = paf_x[ys[i]][xs[i]] * vx + paf_y[ys[i]][xs[i]] * vy;
        if (sub_score > kLocalPafThreshold) {
            score += sub_score;
            count += 1;
        }
    }
    result[0] = score;
    result[1] = count;
    return APP_ERR_OK;
}


APP_ERROR MxpiOpenposePlugin::ScoreSkeletons(int part_idx1, int part_idx2,
                                        const std::vector<int> &part1_x, const std::vector<int> &part1_y,
                                        const std::vector<int> &part2_x, const std::vector<int> &part2_y,
                                        const std::vector<std::vector<float> > &paf_x, const std::vector<std::vector<float> > &paf_y,
                                        const std::vector<std::vector<std::vector<float> > > &keypoint_heatmap,
                                        float rescale_height, float rescale_width,
                                        std::vector<PartPair> &connections) {
    int x1, y1, x2, y2;
    std::vector<PartPair> connection_temp {};
    std::vector<int> result {0, 0};
    for (int i = 0; i < part1_x.size(); i++) {
        x1 = part1_x[i];
        y1 = part1_y[i];
        for (int j = 0; j < part2_x.size(); j++) {
            x2 = part2_x[j];
            y2 = part2_y[j];
            OneSkeletonScore(x1, y1, x2, y2, paf_x, paf_y, result);
            if (result[1] < kPafCountThreshold || result[0] < 0.0)
                continue;
            PartPair skeleton;
            skeleton.score = result[0];
            skeleton.part_idx1 = part_idx1;
            skeleton.part_idx2 = part_idx2;
            skeleton.idx1 = i;
            skeleton.idx2 = j;
            skeleton.coord1.push_back(x1 * rescale_width);
            skeleton.coord1.push_back(y1 * rescale_height);
            skeleton.coord2.push_back(x2 * rescale_width);
            skeleton.coord2.push_back(y2 * rescale_height);
            skeleton.score1 = keypoint_heatmap[part_idx1][y1][x1];
            skeleton.score2 = keypoint_heatmap[part_idx2][y2][x2];
            connection_temp.push_back(skeleton);
        }
    }
    std::vector<int> used_idx1 {};
    std::vector<int> used_idx2 {};
    std::sort(connection_temp.begin(), connection_temp.end(), GreaterSort);
    for (int i = 0; i < connection_temp.size(); i++) {
       PartPair candidate = connection_temp[i];
       if (std::find(used_idx1.begin(), used_idx1.end(), candidate.idx1) != used_idx1.end() || std::find(used_idx2.begin(), used_idx2.end(), candidate.idx2) != used_idx2.end())
           continue;
       connections.push_back(candidate);
       used_idx1.push_back(candidate.idx1);
       used_idx2.push_back(candidate.idx2);
    }
}

APP_ERROR MxpiOpenposePlugin::GenerateSkeletons(const std::vector<std::vector<std::vector<float> > > &paf_heatmap,
                                            const std::vector<std::vector<std::vector<float> > > &keypoint_heatmap,
                                            const std::vector<std::vector<int> > &x_coor,
                                            const std::vector<std::vector<int> > &y_coor,
                                            std::vector<PartPair> &connections) {
    for (int i = 0; i < kNumBodyParts + 1; i++) {
        int coco_skeleton_idx1 = kPoseBodyPartSkeletons[2 * i];
        int coco_skeleton_idx2 = kPoseBodyPartSkeletons[2 * i + 1];
        int paf_x_idx = kPoseMapIndex[2 * i];
        int paf_y_idx = kPoseMapIndex[2 * i + 1];
        // return value
        ScoreSkeletons(
                coco_skeleton_idx1, coco_skeleton_idx2,
                x_coor[coco_skeleton_idx1], y_coor[coco_skeleton_idx1],
                x_coor[coco_skeleton_idx2], y_coor[coco_skeleton_idx2],
                paf_heatmap[paf_x_idx], paf_heatmap[paf_y_idx],
                keypoint_heatmap, 1.0 / keypoint_heatmap[0].size(), 1.0 / keypoint_heatmap[0][0].size(), connections);
    }
}


APP_ERROR MxpiOpenposePlugin::Init(std::map<std::string, std::shared_ptr<void>>& config_param_map)
{
    LogInfo << "MxpiOpenposePlugin::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parent_name_prop_sptr = std::static_pointer_cast<string>(config_param_map["dataSource"]);
    parentName_ = *parent_name_prop_sptr.get();
    std::shared_ptr<string> description_message_pro_sptr =
            std::static_pointer_cast<string>(config_param_map["descriptionMessage"]);
    descriptionMessage_ = *description_message_pro_sptr.get();
    return APP_ERR_OK;
}

APP_ERROR MxpiOpenposePlugin::DeInit()
{
    LogInfo << "MxpiOpenposePlugin::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR MxpiOpenposePlugin::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string plugin_name,
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

APP_ERROR MxpiOpenposePlugin::MergeParts(std::vector<PartPair> &current_person, PartPair current_pair, std::shared_ptr<bool> &result)
{
    for (int i = 0; i < current_person.size(); i++) {
        if (current_pair.part_idx1 == current_person[i].part_idx1) {
            if (current_pair.idx1 == current_person[i].idx1) {
                *result = true;
                return APP_ERR_OK;
            }
        }
        else if (current_pair.part_idx1 == current_person[i].part_idx2) {
            if (current_pair.idx1 == current_person[i].idx2) {
                *result = true;
                return APP_ERR_OK;
            }
        }
        else if (current_pair.part_idx2 == current_person[i].part_idx1) {
            if (current_pair.idx2 == current_person[i].idx1) {
                *result = true;
                return APP_ERR_OK;
            }
        }
        else if (current_pair.part_idx2 == current_person[i].part_idx2) {
            if (current_pair.idx2 == current_person[i].idx2) {
                *result = true;
                return APP_ERR_OK;
            }
        }
    }
    *result = false;
    return APP_ERR_OK;
}

APP_ERROR MxpiOpenposePlugin::NextPersonStart(std::vector<int> &flags, std::shared_ptr<int> &next_idx) {
    for (int i = 0; i < flags.size(); i++) {
        if (flags[i] == -1) {
            *next_idx = i;
            return APP_ERR_OK;
        }
    }
    *next_idx = flags.size();
    return APP_ERR_OK;
}

APP_ERROR MxpiOpenposePlugin::AssemblePerson(std::vector <PartPair> &connections,
                                            mxpiopenposeproto::MxpiPersonList& dst_mxpi_person_list)
{
    int part_num = connections.size();
    std::vector<std::vector<PartPair> > person_list {};
    std::vector<int> person_flags(part_num, -1);
    bool find_new_part = true;
    int person_id = 0;
    int next_idx;
    auto next_idx_ptr = std::make_shared<int>(0);
    next_idx = *next_idx_ptr;
    while (next_idx < part_num - 1) {
        LogInfo << "Current Person: " << person_id;
        std::vector<PartPair> current_person {};
        current_person.push_back(connections[next_idx]);
        person_flags[next_idx] = person_id;
        auto find_new_part = std::make_shared<bool>(false);
        for (int i = next_idx + 1; i < part_num; i++) {
            if (person_flags[i] == -1) {
                MergeParts(current_person, connections[i], find_new_part);
                if (*find_new_part) {
                    current_person.push_back(connections[i]);
                    person_flags[i] = person_id;
                    i = next_idx;
                    *find_new_part = false;
                }
            }
        }
        person_list.push_back(current_person);
        person_id += 1;
        NextPersonStart(person_flags, next_idx_ptr);
        next_idx = *next_idx_ptr;
    }
    LogInfo << "Assemble: " << person_list.size() << " person";
    LogInfo << "Prepare MxpiPersonList data";
    for (int k = 0; k < person_list.size(); k ++) {
        LogInfo << "Add person: " << k;
        auto mxpi_person_ptr = dst_mxpi_person_list.add_personinfovec();
        mxpiopenposeproto::MxpiMetaHeader* dst_person_mxpi_metaheader_list = mxpi_person_ptr->add_headervec();
        dst_person_mxpi_metaheader_list->set_datasource(parentName_);
        dst_person_mxpi_metaheader_list->set_memberid(0);
        for (int j = 0; j < person_list[k].size(); j ++) {
            PartPair skeleton = person_list[k][j];
            auto mxpi_skeleton_ptr = mxpi_person_ptr->add_skeletoninfovec();
            mxpi_skeleton_ptr->set_cocoskeletonindex1(skeleton.part_idx1);
            mxpi_skeleton_ptr->set_cocoskeletonindex2(skeleton.part_idx2);
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

APP_ERROR MxpiOpenposePlugin::GeneratePersonList(const MxpiTensorPackageList src_mxpi_tensor_package,
                                                 mxpiopenposeproto::MxpiPersonList& dst_mxpi_person_list)
{
    // Get Tensor
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(src_mxpi_tensor_package, tensors);
    int channel_keypoint, channel_paf, height, width;
    if (tensors.size() == 0) {
        return APP_ERR_COMM_FAILURE;
    }
    else if (tensors.size() == 1) {
        // tensorflow model
        auto shape = tensors[0].GetShape();
        channel_keypoint = kNumBodyParts + 1;
        channel_paf = shape[3] - kNumBodyParts - 1;
        height = shape[1];
        width = shape[2];
    }
    else if (tensors.size() == 2) {
        // caffe model
        auto shape = tensors[1].GetShape();
        channel_keypoint = shape[1];
        height = shape[2];
        width = shape[3];
        auto shape_p = tensors[0].GetShape();
        channel_paf = shape_p[1];
    }
    else {
        LogInfo << "Unsupported type of model";
        return APP_ERR_COMM_FAILURE;
    }
    std::vector<std::vector<std::vector<float> > > keypoint_heatmap(channel_keypoint, std::vector<std::vector<float> >(height, std::vector<float>(width, 0)));
    std::vector<std::vector<std::vector<float> > > paf_heatmap(channel_paf, std::vector<std::vector<float> >(height, std::vector<float>(width, 0)));
    ReadDataFromTensorCaffe(tensors, keypoint_heatmap, paf_heatmap, channel_keypoint, channel_paf, height, width);

    // Extract Candidate Keypoints
    std::vector<std::vector<int> > x_coor {};
    std::vector<std::vector<int> > y_coor {};
    ExtractKeyPoints(keypoint_heatmap, x_coor, y_coor, channel_keypoint, height, width);

    // Extract Candidate Skeletons
    std::vector<PartPair> connections {};
    GenerateSkeletons(paf_heatmap, keypoint_heatmap, x_coor, y_coor, connections);

    // Generate person from candidate skeletons
    AssemblePerson(connections, dst_mxpi_person_list);
    return APP_ERR_OK;
}

APP_ERROR MxpiOpenposePlugin::Process(std::vector<MxpiBuffer*>& mxpi_buffer)
{
    LogInfo << "MxpiOpenposePlugin::Process start";
    MxpiBuffer* buffer = mxpi_buffer[0];
    MxpiMetadataManager mxpi_metadata_manager(*buffer);
    MxpiErrorInfo mxpi_error_info;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpi_metadata_manager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) << "MxpiOpenposePlugin process is not implemented";
        mxpi_error_info.ret = APP_ERR_COMM_FAILURE;
        mxpi_error_info.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpi_error_info);
        LogError << "MxpiOpenposePlugin process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the data from buffer
    shared_ptr<void> metadata = mxpi_metadata_manager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
        mxpi_error_info.ret = APP_ERR_METADATA_IS_NULL;
        mxpi_error_info.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpi_error_info);
        return APP_ERR_METADATA_IS_NULL; // self define the error code
    }
    // check the proto struct name
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != kSampleKey) {
        ErrorInfo_ << GetError(APP_ERR_PROTOBUF_NAME_MISMATCH, pluginName_)
                   << "Proto struct name is not MxpiTensorPackageList, failed";
        mxpi_error_info.ret = APP_ERR_PROTOBUF_NAME_MISMATCH;
        mxpi_error_info.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpi_error_info);
        return APP_ERR_PROTOBUF_NAME_MISMATCH; // self define the error code
    }

    // Generate output
    shared_ptr<MxpiTensorPackageList> src_mxpi_tensor_packageListSptr = static_pointer_cast<MxpiTensorPackageList>(metadata);
    shared_ptr<mxpiopenposeproto::MxpiPersonList> dst_mxpi_person_listSptr = make_shared<mxpiopenposeproto::MxpiPersonList>();
    APP_ERROR ret = GeneratePersonList(*src_mxpi_tensor_packageListSptr, *dst_mxpi_person_listSptr);
    LogInfo << "DSTMxpiPersonListPtr shape: " << dst_mxpi_person_listSptr->personinfovec_size();
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiOpenposePlugin get skeleton information failed.";
        mxpi_error_info.ret = ret;
        mxpi_error_info.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpi_error_info);
        return ret;
    }
    ret = mxpi_metadata_manager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dst_mxpi_person_listSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiOpenposePlugin add metadata failed.";
        mxpi_error_info.ret = ret;
        mxpi_error_info.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpi_error_info);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "MxpiOpenposePlugin::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> MxpiOpenposePlugin::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parent_name_pro_sptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
            STRING, "dataSource", "name", "the name of previous plugin", "mxpi_modelinfer0", "NULL", "NULL"});
    auto description_message_pro_sptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>{
            STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is MxpiOpenposePlugin", "NULL", "NULL"});
    properties.push_back(parent_name_pro_sptr);
    properties.push_back(description_message_pro_sptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiOpenposePlugin)
