#!/usr/bin/env python
# coding=utf-8
from os import pipe
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, \
    MxDataInput, MxProtobufIn, InProtobufVector, MxBufferInput, \
    MxMetadataInput, MetadataInputVector, StringVector

if __name__ == '__main__':
    # type1:SendData    type2:SendProtobuf
    INTERFACE_TYPE = 1

    # init stream manager
    stream_mgr_api = StreamManagerApi()
    ret = stream_mgr_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" %str(ret))
        exit()
    
    # creat streams by pipeline file
    with open("./pipeSample.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_mgr_api.CreateMultipleStream(pipeline_str)
    if ret != 0:
        print("Failed to create stream, ret=%s" %str(ret))
        exit()    

    # Construct the input of the stream
    data_input = MxDataInput()
    with open("test.jpg", 'rb') as f:
        data_input.data = f.read()

    stream_name = b'pipeSample'
    inplugin_id = 0
    elment_name = b'appsrc0'

    if INTERFACE_TYPE == 1:
        # streamName: bytes, elementName: bytes, data_input: MxDataInput
        # Input data to a specified stream based on streamName

        # build senddata data source
        frame_info = MxpiDataType.MxpiFrameInfo()
        frame_info.frameId = 0
        frame_info.channelId = 0

        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionData.dataStr = data_input.data

        buffer_input = MxBufferInput()
        buffer_input.mxpiFrameInfo = frame_info.SerializeToString()
        buffer_input.mxpiVisionInfo = vision_vec.SerializeToString()
        buffer_input.data = data_input.data

        metedata_input = MxMetadataInput()
        metedata_input.dataSource = elment_name
        metedata_input.dataType = b"MxTools.MxpiVisionList"
        metedata_input.serializedMetadaata = vision_list.SerializeToString()

        metedata_vec = MetadataInputVector()
        metedata_vec.push_back(metedata_input)

        error_code = stream_mgr_api.\
            SendData(stream_name, elment_name, metedata_vec, buffer_input)

        if error_code < 0:
            print("Failed to send data to stream.")
            exit()

        data_source_vector = StringVector()
        data_source_vector.push_back(elment_name)

        infer_result = stream_mgr_api.\
            GetResult(stream_name, b'appsink0', data_source_vector)

        # print the infer result
        if (infer_result.errorCode != 0):
            print("GetResult failed")
            exit()

        if (infer_result.bufferOutput.data is None):
            print("bufferOutput nullptr")
            exit()

        print("result: {}".format(infer_result.bufferOutput.data.decode()))

    elif INTERFACE_TYPE == 2:
        # streanName: bytes, inPluginId: int, protobufVec: list
        # Input data to specified stream base on inplugin_id

        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionData.dataStr = data_input.data

        protobuf = MxProtobufIn()
        protobuf.key = elment_name
        protobuf.type = b'MxTools.MxpiVisionList'
        protobuf.protobuf = vision_list.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)

        error_code = stream_mgr_api. \
            SendProtobuf(stream_name, inplugin_id, protobuf_vec)
        if error_code < 0:
            print("Failed to send data to stream.")
            exit()

        key_vec = StringVector()
        key_vec.push_back(elment_name)
        infer_result = stream_mgr_api. \
            GetProtobuf(stream_name, inplugin_id, key_vec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("infer_result error. \
                errorCode=%d" % (infer_result[0].errorCode))
            exit()

        # print the infer result
        print("GetProtobuf errorCode=%d" % (infer_result[0].errorCode))
        print("KEY: {}".format(str(infer_result[0].messageName)))

        result_visionlist = MxpiDataType.MxpiVisionList()
        result_visionlist.ParseFromString(infer_result[0].messageBuf)
        print("result: {}".format(
            result_visionlist.visionVec[0].visionData.dataStr))

    # destroy streams
    stream_mgr_api.DestroyAllStreams()