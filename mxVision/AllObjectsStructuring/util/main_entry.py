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

import numpy as np

from retrieval.register import Map, Collector, Canvas, parse_feature
from util.multi_process import MultiprocessingQueue, MultiprocessingDictQueue
from util.arguments import get_args_for_all_object_structurization
from util.pipeline import Pipeline
from main_pipeline.main_pipeline import MainPipeline


class AllObjectStructurization:
    def __init__(self, display):
        self.display = display
        self.index_little = None
        self.index_large = None
        self.idx2face_little = None
        self.face_register_pipeline = None
        self.idx2face_large = Map(multi_flag=True)
        self.queue_obj = MultiprocessingQueue()
        self.args = get_args_for_all_object_structurization()
        self.infer_result_queue_by_channel = MultiprocessingDictQueue(
            self.args.main_pipeline_channel_count)
        self.stream_bbox_queue_by_channel = MultiprocessingDictQueue(
            self.args.main_pipeline_channel_count)
        self.main()

    def get_index(self):
        import faiss
        from retrieval.feature_retrieval import Searcher

        d = self.args.index_vector_dimension  # vector dims
        quantizer = faiss.IndexFlatL2(d)

        self.index_little = Searcher(
            mode="normal_flat",
            device_id=self.args.index_little_device_ids,
            quantizer=quantizer,
            d=d,
            nlist=self.args.index_cluster_count)
        self.index_large = Searcher(mode="ivf_sq",
                                    device_id=self.args.index_large_device_ids,
                                    quantizer=quantizer,
                                    d=d,
                                    nlist=self.args.index_cluster_count)

        try:
            self.index_little.load(self.args.index_loading_path_little)
            self.idx2face_little = Map(self.args.idx2face_name_map_path)
        except NotADirectoryError:
            self.create_little_index()

        try:
            self.index_large.load(self.args.index_loading_path_large)
        except NotADirectoryError:
            self.create_large_index()

    def create_little_index(self):
        self.idx2face_little = Map()
        self.register_faces()
        self.index_little.save(self.args.index_loading_path_little)
        self.idx2face_little.save(self.args.idx2face_name_map_path)

    def create_large_index(self):
        np.random.seed(1234)
        xb = np.random.random(
            (self.args.index_base_size,
             self.args.index_vector_dimension)).astype('float32')
        self.index_large.train(xb)
        self.index_large.save(self.args.index_loading_path_large)

    def register_faces(self):
        self.face_register_pipeline = Pipeline(
            pipeline_cfg_file=self.args.face_feature_pipeline_path,
            stream_name=self.args.face_feature_pipeline_name,
            in_plugin_id=0)
        canvas_width, canvas_height = self.args.canvas_size
        collection = Collector(self.args.face_root_path, "face")
        canvas = Canvas(canvas_height, canvas_width)
        idx = 0
        for key, path in collection:
            print(f"{key}: {path}")
            enlarged_img = canvas(path, binary=True)
            ret = self.face_register_pipeline.infer(image_bin=enlarged_img)
            if ret:
                vector = parse_feature(ret)
                if vector:
                    mtx = np.asarray(vector, dtype=np.float32)[np.newaxis, :]
                    self.index_little.add(mtx, idx)
                    self.idx2face_little[str(idx)] = key
                    idx += 1

    def main(self):
        if not self.args.main_pipeline_only:
            self.get_index()

        main_pipeline = MainPipeline(self.args, self.queue_obj, self.stream_bbox_queue_by_channel)
        feature_retrieval = RegisterAndRetrive(self.args, self.index_little,
                                               self.index_large,
                                               self.idx2face_little,
                                               self.idx2face_large,
                                               self.queue_obj,
                                               self.infer_result_queue_by_channel)
        display = self.display(self.args, self.infer_result_queue_by_channel, self.stream_bbox_queue_by_channel)

        try:
            main_pipeline.start()
            display.start()
            feature_retrieval.run()
        except KeyboardInterrupt:
            if main_pipeline.is_alive():
                main_pipeline.kill()
            if display.is_alive():
                display.kill()
            print("Stop AllObjectsStructuring successfully.")


class RegisterAndRetrive:
    def __init__(self, args, index_little, index_large, idx2face_little,
                 idx2face_large, queue_obj, queue_display):
        self.args = args
        self.index_little = index_little
        self.index_large = index_large
        self.idx2face_little = idx2face_little
        self.idx2face_large = idx2face_large
        self.queue_obj = queue_obj
        self.queue_display = queue_display

    def run(self):
        while True:
            obj_dict = self.queue_obj.get()
            channel_id = obj_dict.get("channel_id")
            if not isinstance(channel_id, int):
                raise IOError("Channel Id not found.")

            if self.args.main_pipeline_only or \
                    obj_dict.get("object_name") != "face":
                self.queue_display.put(obj_dict, channel_id)
                continue

            feat_vec = obj_dict.get("feature_vector")
            feat_vec = np.asarray(feat_vec, dtype=np.float32)[np.newaxis, :]
            obj_idx = obj_dict.get("object_index")

            # Todo 序号需要转换成字符吗
            idx_large = self.index_large.add(feat_vec)
            self.idx2face_large[idx_large] = obj_idx

            distance, indexes = self.index_little.search(
                feat_vec, self.args.index_topk)
            idx_little = str(indexes[0][0])
            retrieved_key = self.idx2face_little[idx_little]
            obj_dict["retrieved_key"] = retrieved_key
            self.queue_display.put(obj_dict, channel_id)
