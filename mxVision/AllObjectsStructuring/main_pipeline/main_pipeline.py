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
from util.pipeline import Pipeline
from threading import Thread


class MainPipeline(Process):
    def __init__(self, args, queue_obj, stream_bbox_queue_by_channel):
        super().__init__()
        self.args = args
        self.queue_obj = queue_obj
        self.stream_bbox_queue_by_channel = stream_bbox_queue_by_channel

    def run(self):
        out_plugin_id = 0
        out_stream_bbox_plugin_id = 1
        pipeline = Pipeline(pipeline_cfg_file=self.args.main_pipeline_path,
                            stream_name=self.args.main_pipeline_name,
                            out_plugin_id=out_plugin_id,
                            out_stream_plugin_id=out_stream_bbox_plugin_id,
                            keys=self.args.main_keys2fetch,
                            stream_bbox_key=self.args.main_stream_bbox_keys2fetch
                            )

        get_stream_frame_thread = GetStreamAndBBoxData(self.args, pipeline, self.stream_bbox_queue_by_channel)
        get_stream_frame_thread.start()

        while True:
            buffer = pipeline.get_single_stream_ret(
                save_fig=self.args.main_save_fig,
                base64_enc=self.args.main_base64_encode)
            self.queue_obj.put(buffer)


class GetStreamAndBBoxData(Thread):
    def __init__(self, args, pipeline, stream_bbox_queue_by_channel):
        super().__init__()
        self.args = args
        self.pipeline = pipeline
        self.stream_bbox_queue_by_channel = stream_bbox_queue_by_channel

    def run(self):
        while True:
            stream_bbox_data = self.pipeline.get_stream_bbox_data()
            if self.args.display_stream_bbox_data:
                channel_id = stream_bbox_data["channel_id"]
                self.stream_bbox_queue_by_channel.put(stream_bbox_data, channel_id)
