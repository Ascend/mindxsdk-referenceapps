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

from executor.data.image_loader import ImageLoaderDir
from executor.inference import InferenceExecutor
from executor.model.pipeline import Pipeline
from indicator.criterion import PerCharPrecision, FullSequencePrecision
from interface.eval import Evaluator
from test.build import CRNN_TEST_COLLECTION
from utils.parser import crnn_parsing_func
from utils import constants


@CRNN_TEST_COLLECTION.register(constants.UnitName.PIPELINE.value)
def test_pipeline(cfg):
    loading_path = cfg.get("data_loading_path")
    shared_params = dict()
    crnn_data_loader = ImageLoaderDir(loading_path=loading_path,
                                      shared_params=shared_params)

    pipeline_cfg_file_path = cfg.get("pipeline_cfg_path")
    stream_name_str = cfg.get("stream_name")
    pipeline = Pipeline(pipeline_cfg_file=pipeline_cfg_file_path,
                        stream_name=stream_name_str,
                        parser=crnn_parsing_func)

    for image in crnn_data_loader:
        ret = pipeline(image)
        print(ret)


@CRNN_TEST_COLLECTION.register(constants.UnitName.INFERENCE.value)
def test_inference(cfg):
    loading_path = cfg.get("data_loading_path")
    shared_params = dict()
    crnn_data_loader = ImageLoaderDir(loading_path=loading_path,
                                      shared_params=shared_params)

    pipeline_cfg_file_path = cfg.get("pipeline_cfg_path")
    stream_name_str = cfg.get("stream_name")
    pipeline = Pipeline(pipeline_cfg_file=pipeline_cfg_file_path,
                        stream_name=stream_name_str,
                        parser=crnn_parsing_func)

    crnn_inference = InferenceExecutor(pipeline,
                                       data_loader=crnn_data_loader,
                                       name="crnn_inference",
                                       shared_params=shared_params,
                                       verbose=True)
    pred = crnn_inference.execute()
    print(pred)


@CRNN_TEST_COLLECTION.register(constants.UnitName.EVALUATION.value)
def test_evaluation(cfg):
    loading_path = cfg.get("data_loading_path")
    shared_params = dict()
    crnn_data_loader = ImageLoaderDir(loading_path=loading_path,
                                      shared_params=shared_params)

    pipeline_cfg_file_path = cfg.get("pipeline_cfg_path")
    stream_name_str = cfg.get("stream_name")
    pipeline = Pipeline(pipeline_cfg_file=pipeline_cfg_file_path,
                        stream_name=stream_name_str,
                        parser=crnn_parsing_func)

    crnn_inference = InferenceExecutor(pipeline,
                                       data_loader=crnn_data_loader,
                                       name="crnn_inference",
                                       shared_params=shared_params,
                                       verbose=True)
    label_loading_path = cfg.get("label_loading_path")
    criterion_per_char = PerCharPrecision(loading_path=label_loading_path)
    criterion_full_seq = FullSequencePrecision(loading_path=label_loading_path)
    crnn_evaluator = Evaluator([criterion_per_char, criterion_full_seq],
                               inference_executor=crnn_inference)

    eval_score = crnn_evaluator.eval()
    print(f"Eval score is {eval_score}.")
