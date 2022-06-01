import stat
import json
import os
import datetime
import time
import signal
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open("/home/dongyu3/fruit/pipelines/passengerflowdetection.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    streamName = b'passengerflowestimation_pipline'

    frameid = 0
    f = open("result.h264", "ab")
    FRAMENUM = 1200
    startTime = time.time()
    while frameid < FRAMENUM:
        frameid += 1
        infer_result = streamManagerApi.GetResult(streamName, 0, 10000)
        if infer_result.data == b'[1002][Internal error] ':
            print("Error! cannot find video source!")
            break
        f.write(infer_result.data)
    endTime = time.time()
    rate = FRAMENUM/(endTime - startTime)
    print("Average_framerate:", rate)
    f.close()
    streamManagerApi.DestroyAllStreams()
