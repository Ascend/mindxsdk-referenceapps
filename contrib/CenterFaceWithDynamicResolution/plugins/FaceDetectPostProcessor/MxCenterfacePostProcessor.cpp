/*
 * Copyright (c) 2021.Huawei Technologies Co., Ltd. All rights reserved.
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
#include "MxCenterfacePostProcessor.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include <MxBase/Maths/FastMath.h>
#include <iostream>
using namespace std;
namespace MxBase {
APP_ERROR MxCenterfacePostProcessor::Init(const std::string &configPath,
                                          const std::string &labelPath) {
  APP_ERROR ret = APP_ERR_OK;
  std::map<std::string, std::shared_ptr<void>> postConfig;
  if (!configPath.empty())
    postConfig["postProcessConfigPath"] =
        std::make_shared<std::string>(configPath);
  if (!labelPath.empty())
    postConfig["labelPath"] = std::make_shared<std::string>(labelPath);

  ret = Init(postConfig);
  if (ret == APP_ERR_OK) { // Init for this class derived information
    ret = ReadConfigParams();
  }
  return ret;
}

APP_ERROR MxCenterfacePostProcessor::Init(
    const std::map<std::string, std::shared_ptr<void>> &postConfig) {
  APP_ERROR ret = LoadConfigDataAndLabelMap(postConfig);
  if (ret != APP_ERR_OK) {
    LogError << "LoadConfigDataAndLabelMap failed. ret=" << ret;
    return ret;
  }
  ReadConfigParams();
  LogDebug << "End to Init centerface FaceDetectPostProcessor";
  return APP_ERR_OK;
}

APP_ERROR MxCenterfacePostProcessor::Process(
    const std::vector<MxBase::TensorBase> &tensors,
    std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
    const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
  LogDebug << "Start to Process CenterfacePostProcess ...";
  APP_ERROR ret = APP_ERR_OK;
  auto outputs = tensors;
  ret = CheckAndMoveTensors(outputs);
  if (ret != APP_ERR_OK) {
    LogError << "CheckAndMoveTensors failed:" << ret;
    return ret;
  }

  auto shape = outputs[0].GetShape();
  size_t batch_size = shape[0];
  std::vector<void *> featLayerData;
  MxBase::ResizedImageInfo resizeImgInfo;

  for (size_t i = 0; i < batch_size; i++) {
    std::vector<MxBase::ObjectInfo> objInfo;
    featLayerData.reserve(tensors.size());
    std::transform(tensors.begin(), tensors.end(), featLayerData.begin(),
                   [batch_size, i](MxBase::TensorBase tensor) -> void * {
                     return reinterpret_cast<void *>(
                         reinterpret_cast<char *>(tensor.GetBuffer()) +
                         tensor.GetSize() / batch_size * i);
                   });
    resizeImgInfo = resizedImageInfos[i];
    this->Process(featLayerData, objInfo, resizeImgInfo);
    objectInfos.push_back(objInfo);
  }

  return APP_ERR_OK;
}

APP_ERROR MxCenterfacePostProcessor::Process(std::vector<void *> &featLayerData,
                                             std::vector<MxBase::ObjectInfo> &objInfos,
                                             const MxBase::ResizedImageInfo &resizeInfo) {
  ImageInfo imageInfo;
  imageInfo.modelWidth = resizeInfo.widthResize;
  imageInfo.modelHeight = resizeInfo.heightResize;
  imageInfo.imgHeight = resizeInfo.heightOriginal;
  imageInfo.imgWidth = resizeInfo.widthOriginal;
  modelWidth_ = resizeInfo.widthResize / downSample;
  modelHeight_ = resizeInfo.heightResize / downSample;
  std::vector<FaceInfo> faces;
  detect(featLayerData, faces, imageInfo, scoreThresh_, iouThresh_);
  for (int i = 0; i < faces.size(); i++) {
    MxBase::ObjectInfo objectInfo;
    objectInfo.x0 = faces[i].x1;
    objectInfo.x1 = faces[i].x2;
    objectInfo.y0 = faces[i].y1;
    objectInfo.y1 = faces[i].y2;
    objectInfo.confidence = faces[i].score;
    objInfos.push_back(objectInfo);
  }
  return APP_ERR_OK;
}

APP_ERROR MxCenterfacePostProcessor::ReadConfigParams() {
  configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
  configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
  configData_.GetFileValue<int>("NMS_METHOD", numsMethod);
  return APP_ERR_OK;
}

void MxCenterfacePostProcessor::detect(std::vector<void *> &featLayerData,
                                       std::vector<FaceInfo> &faces,
                                       const ImageInfo &imgInfo,
                                       float scoreThresh, float nmsThresh) {
  scale_w = (float)imgInfo.imgWidth / (float)imgInfo.modelWidth;
  scale_h = (float)imgInfo.imgHeight / (float)imgInfo.modelHeight;
  int hotMapIndex = 0;
  int scaleIndex = 1;
  int offsetIndex = 2;
  int landMarksIndex = 3;
  decode((float *)featLayerData[hotMapIndex],
         (float *)featLayerData[scaleIndex],
         (float *)featLayerData[offsetIndex],
         (float *)featLayerData[landMarksIndex], faces, imgInfo, scoreThresh,
         nmsThresh);
  squareBox(faces, imgInfo);
}

void MxCenterfacePostProcessor::decode(float *heatmap, float *scale,
                                       float *offset, float *landmarks,
                                       std::vector<FaceInfo> &faces,
                                       const ImageInfo &imageinfo,
                                       float scoreThresh, float nmsThresh) {
  int spacial_size = modelHeight_ * modelWidth_;

  float *heatmap_ = heatmap;

  float *scale0 = scale;
  float *scale1 = scale0 + spacial_size;

  float *offset0 = offset;
  float *offset1 = offset0 + spacial_size;
  float *lm = landmarks;

  std::vector<int> ids =
      getIds(heatmap_, modelHeight_, modelWidth_, scoreThresh);
  int pairLength = 2;
  int Step = 2;
  for (int i = 0; i < ids.size() / pairLength; i++) {
    int id_h = ids[2 * i];
    int id_w = ids[2 * i + 1];
    int index = id_h * modelWidth_ + id_w;

    float s0 = std::exp(scale0[index]) * 4;
    float s1 = std::exp(scale1[index]) * 4;
    float o0 = offset0[index];
    float o1 = offset1[index];

    float x1 = std::max(0., (id_w + o1 + 0.5) * 4 - s1 / 2);
    float y1 = std::max(0., (id_h + o0 + 0.5) * 4 - s0 / 2);
    float x2 = 0, y2 = 0;
    x1 = std::min(x1, (float)imageinfo.modelWidth);
    y1 = std::min(y1, (float)imageinfo.modelHeight);
    x2 = std::min(x1 + s1, (float)imageinfo.modelWidth);
    y2 = std::min(y1 + s0, (float)imageinfo.modelHeight);

    FaceInfo facebox;
    facebox.x1 = x1;
    facebox.y1 = y1;
    facebox.x2 = x2;
    facebox.y2 = y2;
    facebox.score = heatmap_[index];
    int keyPointNums = 5;
    for (int j = 0; j < keyPointNums; j++) {
      facebox.landmarks[Step * j] =
          x1 + lm[(Step * j + 1) * spacial_size + index] * s1;
      facebox.landmarks[Step * j + 1] =
          y1 + lm[(Step * j) * spacial_size + index] * s0;
    }
    faces.push_back(facebox);
  }

  nms(faces, scoreThresh, iouThresh_, numsMethod);
  int keyPointNums = 5;
  for (int k = 0; k < faces.size(); k++) {
    faces[k].x1 *= scale_w;
    faces[k].y1 *= scale_h;
    faces[k].x2 *= scale_w;
    faces[k].y2 *= scale_h;

    for (int kk = 0; kk < keyPointNums; kk++) {
      faces[k].landmarks[Step * kk] *= scale_w;
      faces[k].landmarks[Step * kk + 1] *= scale_h;
    }
  }
}

std::vector<int> MxCenterfacePostProcessor::getIds(float *heatmap, int h, int w,
                                                   float thresh) {
  std::vector<int> ids;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (heatmap[i * w + j] > thresh) {
        ids.push_back(i);
        ids.push_back(j);
      }
    }
  }
  return ids;
}

void MxCenterfacePostProcessor::squareBox(std::vector<FaceInfo> &faces,
                                          const ImageInfo &imageinfo) {
  float w = 0, h = 0, maxSize = 0;
  float cenx, ceny;
  for (int i = 0; i < faces.size(); i++) {
    w = faces[i].x2 - faces[i].x1;
    h = faces[i].y2 - faces[i].y1;
    int scaleFactor = 2;
    maxSize = std::max(w, h);
    cenx = faces[i].x1 + w / scaleFactor;
    ceny = faces[i].y1 + h / scaleFactor;

    faces[i].x1 = std::max(cenx - maxSize / scaleFactor, 0.f);
    faces[i].y1 = std::max(ceny - maxSize / scaleFactor, 0.f);
    faces[i].x2 =
        std::min(cenx + maxSize / scaleFactor, imageinfo.imgWidth - 1.f);
    faces[i].y2 =
        std::min(ceny + maxSize / scaleFactor, imageinfo.imgHeight - 1.f);
  }
}

void MxCenterfacePostProcessor::nms(std::vector<FaceInfo> &vec_boxs,
                                    float nmsthresh, float iouthresh,
                                    unsigned int method, float sigma) {
  int box_len = vec_boxs.size();
  for (int i = 0; i < box_len; i++) {
    FaceInfo *max_ptr = &vec_boxs[i];
    // get max box
    for (int pos = i + 1; pos < box_len; pos++)
      if (vec_boxs[pos].score > max_ptr->score)
        max_ptr = &vec_boxs[pos];

    // swap ith box with position of max box
    if (max_ptr != &vec_boxs[i])
      std::swap(*max_ptr, vec_boxs[i]);

    max_ptr = &vec_boxs[i];

    for (int pos = i + 1; pos < box_len; pos++) {
      FaceInfo &curr_box = vec_boxs[pos];
      float area =
          (curr_box.x2 - curr_box.x1 + 1) * (curr_box.y2 - curr_box.y1 + 1);
      float iw = std::min(max_ptr->x2, curr_box.x2) -
                 std::max(max_ptr->x1, curr_box.x1) + 1;
      float ih = std::min(max_ptr->y2, curr_box.y2) -
                 std::max(max_ptr->y1, curr_box.y1) + 1;
      if (iw > 0 && ih > 0) {
        float overlaps = iw * ih;
        // iou between max box and detection box
        float iou = overlaps / ((max_ptr->x2 - max_ptr->x1 + 1) *
                                    (max_ptr->y2 - max_ptr->y1 + 1) +
                                area - overlaps);
        float weight = 0;
        if (method == NumsMethod::LINEAR) // linear
          weight = iou > iouthresh ? 1 - iou : 1;
        else if (method == NumsMethod::GAUSSIAN) { // gaussian
          if (sigma == 0) {
            LogError << "tht value of sigma shouldn't be zero";
          }
          weight = std::exp(-(iou * iou) / sigma);
        } else // original NMS
          weight = iou > iouthresh ? 0 : 1;
        // adjust all bbox score after this box
        curr_box.score *= weight;
        // if new confidence less then threshold , swap with last one
        // and shrink this array
        if (curr_box.score < nmsthresh) {
          std::swap(curr_box, vec_boxs[box_len - 1]);
          box_len--;
          pos--;
        }
      }
    }
  }
  vec_boxs.resize(box_len);
}

extern "C" {
std::shared_ptr<MxBase::MxCenterfacePostProcessor> GetObjectInstance() {
  LogInfo << "Begin to get CenterFacePostProcess instance.";
  auto instance = std::make_shared<MxCenterfacePostProcessor>();
  LogInfo << "End to get CenterFacePostProcess instance.";
  return instance;
}
}
} // namespace MxBase
