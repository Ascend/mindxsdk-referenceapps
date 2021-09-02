import signal
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType

def Visualization(img, infer, shape, frameId, channelId):
    """
    :param img: Inference image
    :param infer: Inference result
    :param shape:Size before image padding
    :param frameId:Inference image id
    :param channelId:Channel id of the current inference image
    func: the visualization of the inference result, save the output in the specified folder
    """
    imgLi2 = []
    print(shape)
    # The title of the rectangle
    tl = (round(0.0002 * (shape[0] + shape[1])) + 0.35)
    tf = max(tl - 1, 1)
    # Add the inference results of head to imgLi2
    for bbox0 in infer:
        if bbox0[5] == 'head':
            imgLi2.append(bbox0)
    for bbox1 in imgLi2:
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
        L1 = []
        L1.append(int(bboxes['x0']))
        L1.append(int(bboxes['x1']))
        L1.append(int(bboxes['y0']))
        L1.append(int(bboxes['y1']))
        L1 = np.array(L1, dtype=np.int32)
        # Draw rectangle
        cv2.putText(img, str(bboxes['confidence']), (L1[0], L1[2]), 0, tl, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)
        # rectangle color [255,255,255]
        cv2.rectangle(img, (L1[0], L1[2]), (L1[1], L1[3]), (0, 0, 255), 2)
        if bboxes['trackid'] is not None and bboxes['age'] == 1:
            print("Warning:Not wearing a helmet,InferenceId:{},FrameId:{}".format(bboxes['trackid'], frameId))

    # Save pictures in two ways
    if channelId == 0:
        oringe_imgfile = './output/one/image' + str(channelId) + '-' + str(
            frameId) + '.jpg'
        # Warning result save path
        cv2.imwrite(oringe_imgfile, img)
    else:
        # when channelId equal 1
        oringe_imgfile = './output/two/image' + str(channelId) + '-' + str(
            frameId) + '.jpg'
        cv2.imwrite(oringe_imgfile, img)


def get_inference_data(inference):
    """
    :param inference:output of sdk stream inference
    :return:img0, imgLi1, img0_shape, FrameList0.frameId, FrameList0.channelId
    """

    # add inferennce data into DATA structure
    # Frame information structure
    FrameList0 = MxpiDataType.MxpiFrameInfo()
    FrameList0.ParseFromString(inference[0].messageBuf)
    # Target object structure
    ObjectList = MxpiDataType.MxpiObjectList()
    ObjectList.ParseFromString(inference[1].messageBuf)
    # Get target box information
    ObjectListData = ObjectList.objectVec
    # track structure
    trackLetList = MxpiDataType.MxpiTrackLetList()
    trackLetList.ParseFromString(inference[2].messageBuf)
    # Obtain tracking information
    trackLetData = trackLetList.trackLetVec
    # image structure
    visionList0 = MxpiDataType.MxpiVisionList()
    visionList0.ParseFromString(inference[3].messageBuf)
    visionData0 = visionList0.visionVec[0].visionData.dataStr
    # Get picture information
    visionInfo0 = visionList0.visionVec[0].visionInfo

    # cv2:YUV2BGR
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv = np.frombuffer(visionData0, dtype=np.uint8)
    # reshape
    img_yuv = img_yuv.reshape(visionInfo0.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo0.widthAligned)
    # Color gamut conversion
    img0 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
    # put inference into dict,
    imgLi1 = []
    for k in range(len(ObjectList.objectVec)):
        imgLi = [round(ObjectListData[k].x0, 4), round(ObjectListData[k].x1, 4), round(ObjectListData[k].y0, 4),
                 round(ObjectListData[k].y1, 4),
                 round(ObjectListData[k].classVec[0].confidence, 4), ObjectListData[k].classVec[0].className,
                 trackLetData[k].trackId, trackLetData[k].age]
        imgLi1.append(imgLi)

    # img0_shape is the original image size
    img0_shape = [visionInfo0.heightAligned, visionInfo0.widthAligned]
    # Output the results uniformly through the dictionary
    DictStructure = [img0, imgLi1, img0_shape, FrameList0.frameId, FrameList0.channelId]
    return DictStructure
