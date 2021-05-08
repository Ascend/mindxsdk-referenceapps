#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from multiprocessing.context import Process
from threading import Thread
from multiprocessing import Value


class Display(Process):
    """
    module for displaying
    """

    def __init__(self, args, infer_result_queue_by_channel, stream_bbox_queue_by_channel):
        super().__init__()
        self.args = args
        self.infer_result_queue_by_channel = infer_result_queue_by_channel
        self.stream_bbox_queue_by_channel = stream_bbox_queue_by_channel
        self.channel_switch = [Value('I', 0) for _ in infer_result_queue_by_channel]

    def run(self):
        if self.args.display_stream_bbox_data:
            stream_bbox_processor_list = [
                WebDisplayProcessor(stream_bbox_display_queue, channel_id)
                for channel_id, stream_bbox_display_queue in self.stream_bbox_queue_by_channel
            ]
            for processor in stream_bbox_processor_list:
                processor.start()

        for switch in self.channel_switch:
            switch.value = 1

        infer_result_processor_list = [
            InferResultProcessor(infer_result_display_queue, channel_id, switch)
            for (channel_id, infer_result_display_queue
                 ), switch in
            zip(self.infer_result_queue_by_channel, self.channel_switch)
        ]
        for processor in infer_result_processor_list:
            processor.start()

        for processor in infer_result_processor_list:
            processor.join()


class InferResultProcessor(Thread):
    """
    infer result displaying thread for each channel
    """

    def __init__(self, infer_result_display_queue, channel_id, switch):
        super().__init__()
        self.channel_id = channel_id
        self.infer_result_display_queue = infer_result_display_queue
        self.switch = switch

    def run(self):
        while True:
            ret = self.infer_result_display_queue.get()
            del ret["attribute"]
            del ret["feature_vector"]
            del ret["image_encoded"]
            if ret.get("object"):
                del ret["object"]
            if not self.switch.value:
                continue

            print("output result:")
            print(ret)


class WebDisplayProcessor(Thread):
    """
    stream and bounding box  displaying thread for each channel
    """

    def __init__(self, stream_bbox_display_queue, channel_id):
        super().__init__()
        self.channel_id = channel_id
        self.stream_bbox_display_queue = stream_bbox_display_queue

    def run(self):
        while True:
            ret = self.stream_bbox_display_queue.get()
            del ret["web_display_data_serialized"]
            del ret["web_display_data_dict"]
            print("stream data and bounding box result:")
            print(ret)
