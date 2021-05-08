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
from indicator.criterion import TextDetectEvaluation
from interface.eval import Evaluator
from test.build import CTPN_TEST_COLLECTION
from utils.parser import ctpn_parse_and_save_func
from utils import constants


@CTPN_TEST_COLLECTION.register(constants.UnitName.EVALUATION.value)
def test_evaluation():
    loading_path = "./test/models/icdar2013/data"
    result_path = "./test/models/icdar2013/result"
    result_zip_name = "icdar2013_test.zip"
    shared_params = dict()
    shared_params['result_path'] = result_path
    shared_params['result_zip_name'] = result_zip_name
    icdar_data_loader = ImageLoaderDir(loading_path=loading_path,
                                       shared_params=shared_params)
    pipeline_cfg_file_path = "./test/models/ctpn_single_cv.pipeline"
    stream_name_str = "detection"
    pipeline = Pipeline(pipeline_cfg_file=pipeline_cfg_file_path,
                        stream_name=stream_name_str,
                        parser=ctpn_parse_and_save_func,
                        shared_params=shared_params)

    ctpn_inference = InferenceExecutor([pipeline],
                                       data_loader=icdar_data_loader,
                                       name="ctpn_mindspore_inference",
                                       shared_params=shared_params,
                                       verbose=False)
    ctpn_gt_path = "./test/models/icdar2013/icdar2013_gt.zip"
    criterion_coco = TextDetectEvaluation(loading_path=ctpn_gt_path, result_path=result_path,
                                          result_zip_name=result_zip_name)
    ctpn_evaluator = Evaluator(criterion_coco, inference_executor=ctpn_inference)
    eval_score = ctpn_evaluator.eval()
    print(f"Eval score is {eval_score}.")
