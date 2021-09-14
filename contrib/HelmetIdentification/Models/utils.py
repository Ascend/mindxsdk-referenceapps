# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType

def cv_visualization(img, infer, shape, frame_id, channel_id):
    """
    :param img: Inference image
    :param infer: Inference result
    :param shape:Size before image padding
    :param frame_id:Inference image id
    :param channel_id:Channel id of the current inference image
    func: the visualization of the inference result, save the output in the specified folder
    """
    img_list2 = []
    print(shape)
    # The title of the rectangle
    title_l = (round(0.0002 * (shape[0] + shape[1])) + 0.35)
    title_f = max(title_l - 1, 1)
    # Add the inference results of head to img_list2
    for bbox0 in infer:
        if bbox0[5] == 'head':
            img_list2.append(bbox0)
    for bbox1 in img_list2:
        # Determine whether it is helmet
        bboxes = {'x0': int(bbox1[0]),
                  'x1': int(bbox1[1]),
                  'y0': int(bbox1[2]),
                  'y1': int(bbox1[3]),
                  'confidence': round(bbox1[4], 4),
                  'trackid': int(bbox1[6]),
                  'age': int(bbox1[7])
                  }
        print(bboxes)
        bboxes_list1 = []
        bboxes_list1.append(int(bboxes['x0']))
        bboxes_list1.append(int(bboxes['x1']))
        bboxes_list1.append(int(bboxes['y0']))
        bboxes_list1.append(int(bboxes['y1']))
        bboxes_list1 = np.array(bboxes_list1, dtype=np.int32)
        # Draw rectangle
        cv2.putText(img, str(bboxes['confidence']), (bboxes_list1[0], bboxes_list1[2]), 0, title_l, [225, 255, 255], thickness=title_f,
                    lineType=cv2.LINE_AA)
        # rectangle color [255,255,255]
        cv2.rectangle(img, (bboxes_list1[0], bboxes_list1[2]), (bboxes_list1[1], bboxes_list1[3]), (0, 0, 255), 2)
        if bboxes['trackid'] is not None and bboxes['age'] == 1:
            print("Warning:Not wearing a helmet,InferenceId:{},FrameId:{}".format(bboxes['trackid'], frame_id))

    # Save pictures in two ways
    if channel_id == 0:
        oringe_imgfile = './output/one/image' + str(channel_id) + '-' + str(
            frame_id) + '.jpg'
        # Warning result save path
        cv2.imwrite(oringe_imgfile, img)
    else:
        # when channel_id equal 1
        oringe_imgfile = './output/two/image' + str(channel_id) + '-' + str(
            frame_id) + '.jpg'
        cv2.imwrite(oringe_imgfile, img)


def get_inference_data(inference):
    """
    :param inference:output of sdk stream inference
    :return:img0, img_list1, img0_shape, frame_list0.frameId, frame_list0.channelId
    """

    # add inferennce data into DATA structure
    # Frame information structure
    frame_list0 = MxpiDataType.MxpiFrameInfo()
    frame_list0.ParseFromString(inference[0].messageBuf)
    # Target object structure
    object_list = MxpiDataType.MxpiObjectList()
    object_list.ParseFromString(inference[1].messageBuf)
    # Get target box information
    objectlist_data = object_list.objectVec
    # track structure
    tracklet_list = MxpiDataType.MxpiTrackLetList()
    tracklet_list.ParseFromString(inference[2].messageBuf)
    # Obtain tracking information
    tracklet_data = tracklet_list.trackLetVec
    # image structure
    vision_list0 = MxpiDataType.MxpiVisionList()
    vision_list0.ParseFromString(inference[3].messageBuf)
    vision_data0 = vision_list0.visionVec[0].visionData.dataStr
    # Get picture information
    vision_info0 = vision_list0.visionVec[0].visionInfo

    # cv2 func YUV to BGR
    yuv_bytes_nu = 3
    yuv_bytes_de = 2
    img_yuv = np.frombuffer(vision_data0, dtype=np.uint8)
    # reshape
    img_yuv = img_yuv.reshape(vision_info0.heightAligned * yuv_bytes_nu // yuv_bytes_de, vision_info0.widthAligned)
    # Color gamut conversion
    img0 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
    # put inference into dict,
    img_list1 = []
    for k in range(len(object_list.objectVec)):
        img_list = [round(objectlist_data[k].x0, 4), round(objectlist_data[k].x1, 4), round(objectlist_data[k].y0, 4),
                 round(objectlist_data[k].y1, 4),
                 round(objectlist_data[k].classVec[0].confidence, 4), objectlist_data[k].classVec[0].className,
                 tracklet_data[k].trackId, tracklet_data[k].age]
        img_list1.append(img_list)

    # img0_shape is the original image size
    img0_shape = [vision_info0.heightAligned, vision_info0.widthAligned]
    # Output the results uniformly through the dictionary
    dict_structure = [img0, img_list1, img0_shape, frame_list0.frameId, frame_list0.channelId]
    return dict_structure
