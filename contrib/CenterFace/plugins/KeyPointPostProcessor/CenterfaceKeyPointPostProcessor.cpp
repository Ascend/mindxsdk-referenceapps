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
#include "CenterfaceKeyPointPostProcessor.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace std;
namespace MxBase {
APP_ERROR CenterfaceKeyPointPostProcessor::Init(
    const std::map<std::string, std::shared_ptr<void>> &postConfig) {
  APP_ERROR ret = KeypointPostProcessBase::Init(postConfig);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Fail to superInit in KeypointPostProcessBase.";
    return ret;
  }
  configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
  configData_.GetFileValue<float>("IOU_THRESH", nmsThresh_);
  configData_.GetFileValue<int>("NMS_METHOD", nmsMethod);
  LogDebug << "End to Init CenterfaceKeyPointPostProcessor";
  return APP_ERR_OK;
}

APP_ERROR CenterfaceKeyPointPostProcessor::Process(
    const std::vector<MxBase::TensorBase> &tensors,
    std::vector<std::vector<KeyPointDetectionInfo>> &keyPointInfos,
    const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
  LogDebug << "Start to Process CenterfaceKeyPointPostProcessor ...";
  auto outputs = tensors;
  APP_ERROR ret = CheckAndMoveTensors(outputs);
  if (ret != APP_ERR_OK) {
    LogError << "CheckAndMoveTensors failed:" << ret;
    return ret;
  }

  auto shape = outputs[0].GetShape();
  size_t batch_size = shape[0];
  std::vector<void *> featLayerData;
  MxBase::ResizedImageInfo resizeImgInfo;

  for (size_t i = 0; i < batch_size; i++) {
    std::vector<MxBase::KeyPointDetectionInfo> keyPointInfo;
    featLayerData.reserve(tensors.size());
    std::transform(tensors.begin(), tensors.end(), featLayerData.begin(),
                   [batch_size, i](MxBase::TensorBase tensor) -> void * {
                     return reinterpret_cast<void *>(
                         reinterpret_cast<char *>(tensor.GetBuffer()) +
                         tensor.GetSize() / batch_size * i);
                   });
    resizeImgInfo = resizedImageInfos[i];
    ret = this->ProcessOnePicture(featLayerData, keyPointInfo, resizeImgInfo);
    if (ret != APP_ERR_OK) {
      LogError << "Postprocessing failed:" << ret;
      return ret;
    }
    keyPointInfos.push_back(keyPointInfo);
  }
  return APP_ERR_OK;
}

APP_ERROR CenterfaceKeyPointPostProcessor::ProcessOnePicture(
    std::vector<void *> &featLayerData,
    std::vector<KeyPointDetectionInfo> &keyPointInfos,
    const MxBase::ResizedImageInfo &resizeInfo) {
  ImageInfo imageInfo;
  imageInfo.modelWidth = resizeInfo.widthResize;
  imageInfo.modelHeight = resizeInfo.heightResize;
  imageInfo.imgHeight = resizeInfo.heightOriginal;
  imageInfo.imgWidth = resizeInfo.widthOriginal;
  modelWidth_ = resizeInfo.widthResize / DOWN_SAMPLE;
  modelHeight_ = resizeInfo.heightResize / DOWN_SAMPLE;
  std::vector<FaceInfo> faces;
  APP_ERROR ret = detect(featLayerData, faces, imageInfo);
  if (ret != APP_ERR_OK) {
    LogInfo << GetError(ret) << "fail to detect  face.";
    return ret;
  }
  int keyPointNums = 5;
  for (int i = 0; i < faces.size(); i++) {
    MxBase::KeyPointDetectionInfo keypointInfo;
    for (int j = 0; j < keyPointNums; j++) {
      vector<float> temp = {faces[i].landmarks[2 * j],
                            faces[i].landmarks[2 * j + 1]};
      keypointInfo.keyPointMap[j] = temp;
    }
    keyPointInfos.push_back(keypointInfo);
  }
  return APP_ERR_OK;
}

APP_ERROR CenterfaceKeyPointPostProcessor::detect(std::vector<void *> &featLayerData,
                                                  std::vector<FaceInfo> &faces,
                                                  const ImageInfo &imgInfo) {
  scale_w = (float)imgInfo.imgWidth / (float)imgInfo.modelWidth;
  scale_h = (float)imgInfo.imgHeight / (float)imgInfo.modelHeight;
  int hotMapIndex = 0;
  int scaleIndex = 1;
  int offsetIndex = 2;
  int landMarksIndex = 3;
  APP_ERROR ret = decode((float *)featLayerData[hotMapIndex],
                         (float *)featLayerData[scaleIndex],
                         (float *)featLayerData[offsetIndex],
                         (float *)featLayerData[landMarksIndex], faces, imgInfo);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "fail to decode the results of model inference.";
    return ret;
  }
  squareBox(faces, imgInfo);
  return APP_ERR_OK;
}

APP_ERROR CenterfaceKeyPointPostProcessor::decode(float *heatmap, float *scale,
                                                  float *offset, float *landmarks,
                                                  std::vector<FaceInfo> &faces,
                                                  const ImageInfo &imageinfo) {
  int spacial_size = modelHeight_ * modelWidth_;

  float *heatmap_ = heatmap;

  float *scale0 = scale;
  float *scale1 = scale0 + spacial_size;

  float *offset0 = offset;
  float *offset1 = offset0 + spacial_size;
  float *lm = landmarks;
  std::vector<int> ids;
  try {
    ids = getIds(heatmap_, modelHeight_, modelWidth_);
  } catch (exception e) {
    return APP_ERR_COMM_FAILURE;
  }

  int pairLength = 2;
  int Step = 2;
  FaceInfo facebox;
  try {
    for (int i = 0; i < ids.size() / pairLength; i++) {
      int id_h = ids[2 * i];
      int id_w = ids[2 * i + 1];
      int index = id_h * modelWidth_ + id_w;

      float s0 = std::exp(scale0[index]) * DOWN_SAMPLE;
      float s1 = std::exp(scale1[index]) * DOWN_SAMPLE;
      float o0 = offset0[index];
      float o1 = offset1[index];

      float x1 = std::max(0., (id_w + o1 + 0.5) * DOWN_SAMPLE - s1 / SCALE_FACTOR);
      float y1 = std::max(0., (id_h + o0 + 0.5) * DOWN_SAMPLE - s0 / SCALE_FACTOR);

      float x2 = 0, y2 = 0;
      x1 = std::min(x1, (float) imageinfo.modelWidth);
      y1 = std::min(y1, (float) imageinfo.modelHeight);
      x2 = std::min(x1 + s1, (float) imageinfo.modelWidth);
      y2 = std::min(y1 + s0, (float) imageinfo.modelHeight);

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
  } catch(exception e) {
    return APP_ERR_COMM_OUT_OF_RANGE;
  }

  nms(faces, nmsMethod);
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
  return APP_ERR_OK;
}

std::vector<int> CenterfaceKeyPointPostProcessor::getIds(float *heatmap, int h, int w) {
  std::vector<int> ids;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (heatmap[i * w + j] > scoreThresh_) {
        ids.push_back(i);
        ids.push_back(j);
      }
    }
  }
  return ids;
}

void CenterfaceKeyPointPostProcessor::squareBox(std::vector<FaceInfo> &faces,
                                                const ImageInfo &imageinfo) {
  float w = 0, h = 0, maxSize = 0;
  float cenx = 0, ceny = 0;
  for (int i = 0; i < faces.size(); i++) {
    w = faces[i].x2 - faces[i].x1;
    h = faces[i].y2 - faces[i].y1;
    maxSize = std::max(w, h);
    cenx = faces[i].x1 + w / SCALE_FACTOR;
    ceny = faces[i].y1 + h / SCALE_FACTOR;

    faces[i].x1 = std::max(cenx - maxSize / SCALE_FACTOR, 0.f);
    faces[i].y1 = std::max(ceny - maxSize / SCALE_FACTOR, 0.f);
    faces[i].x2 =
            std::min(cenx + maxSize / SCALE_FACTOR, imageinfo.imgWidth - 1.f);
    faces[i].y2 =
            std::min(ceny + maxSize / SCALE_FACTOR, imageinfo.imgHeight - 1.f);
  }
}

// 根据Nms方法获取weight
float CenterfaceKeyPointPostProcessor::GetNmsWeight(float iou, float sigma, int method) {
  float weight = 0;
  if (method == NmsMethod::LINEAR) // linear
    weight = iou > nmsThresh_ ? 1 - iou : 1;
  else if (method == NmsMethod::GAUSSIAN) { // gaussian
    if (sigma == 0) {
      LogError << "tht value of sigma shouldn't be zero";
    }
    weight = std::exp(-(iou * iou) / sigma);
  } else // original NMS
    weight = iou > nmsThresh_ ? 0 : 1;
  return weight;
}

// 计算两个方框的IOU
float CenterfaceKeyPointPostProcessor::GetIou(FaceInfo &curr_box, FaceInfo *max_ptr, float overlaps) {
  float area =
          (curr_box.x2 - curr_box.x1 + 1) * (curr_box.y2 - curr_box.y1 + 1);
  // iou between max box and detection box
  float iou = overlaps / ((max_ptr->x2 - max_ptr->x1 + 1) *
                          (max_ptr->y2 - max_ptr->y1 + 1) +
                          area - overlaps);
  return iou;
}

void CenterfaceKeyPointPostProcessor::nms(std::vector<FaceInfo> &vec_boxs,
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

      float iw = std::min(max_ptr->x2, curr_box.x2) -
                 std::max(max_ptr->x1, curr_box.x1) + 1;
      float ih = std::min(max_ptr->y2, curr_box.y2) -
                 std::max(max_ptr->y1, curr_box.y1) + 1;
      if (iw > 0 && ih > 0) {
        float overlaps = iw * ih;
        float iou = GetIou(curr_box, max_ptr, overlaps);
        float weight = GetNmsWeight(iou, sigma, method);
        // adjust all bbox score after this box
        curr_box.score *= weight;
        // if new confidence less then threshold , swap with last one
        // and shrink this array
        if (curr_box.score < scoreThresh_) {
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
std::shared_ptr<MxBase::CenterfaceKeyPointPostProcessor> GetKeypointInstance() {
  LogInfo << "Begin to get CenterfaceKeyPointPostProcessor instance.";
  auto instance = std::make_shared<CenterfaceKeyPointPostProcessor>();
  LogInfo << "End to get CenterfaceKeyPointPostProcessor instance.";
  return instance;
}
}
} // namespace MxBase