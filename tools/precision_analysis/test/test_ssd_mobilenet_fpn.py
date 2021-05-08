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
from indicator.criterion import COCOEvaluation
from interface.eval import Evaluator
from test.build import SSD_MOBILENET_FPN_TEST_COLLECTION
from utils.parser import ssd_mobilenet_fpn_parsing_func, label_mapping_for_coco
from utils import constants


def transform_coco_cat_id(annotations):
    for ann in annotations:
        ann["category_id"] = label_mapping_for_coco(ann["category_id"])

    return annotations


@SSD_MOBILENET_FPN_TEST_COLLECTION.register(constants.UnitName.PIPELINE.value)
def test_pipeline(cfg):
    loading_path = cfg.get("data_loading_path")
    shared_params = dict()
    coco_data_loader = ImageLoaderDir(loading_path=loading_path,
                                      shared_params=shared_params)

    pipeline_cfg_file_path = cfg.get("pipeline_cfg_path")
    stream_name_str = cfg.get("stream_name")
    pipeline = Pipeline(pipeline_cfg_file=pipeline_cfg_file_path,
                        stream_name=stream_name_str,
                        parser=ssd_mobilenet_fpn_parsing_func,
                        shared_params=shared_params)

    for image in coco_data_loader:
        ret = pipeline(image)
        ret = transform_coco_cat_id(ret)
        print(ret)

    print("Done !")


@SSD_MOBILENET_FPN_TEST_COLLECTION.register(constants.UnitName.INFERENCE.value)
def test_inference(cfg):
    loading_path = cfg.get("data_loading_path")
    shared_params = dict()
    coco_data_loader = ImageLoaderDir(loading_path=loading_path,
                                      shared_params=shared_params)

    pipeline_cfg_file_path = cfg.get("pipeline_cfg_path")
    stream_name_str = cfg.get("stream_name")
    pipeline = Pipeline(pipeline_cfg_file=pipeline_cfg_file_path,
                        stream_name=stream_name_str,
                        parser=ssd_mobilenet_fpn_parsing_func,
                        shared_params=shared_params)

    ssd_inference = InferenceExecutor([pipeline, transform_coco_cat_id],
                                      data_loader=coco_data_loader,
                                      name="ssd_mobilenet_fpn_inference",
                                      shared_params=shared_params,
                                      verbose=True)
    pred = ssd_inference.execute()
    print(pred)


@SSD_MOBILENET_FPN_TEST_COLLECTION.register(constants.UnitName.EVALUATION.value)
def test_evaluation(cfg):
    loading_path = cfg.get("data_loading_path")
    shared_params = dict()
    coco_data_loader = ImageLoaderDir(loading_path=loading_path,
                                      shared_params=shared_params)

    pipeline_cfg_file_path = cfg.get("pipeline_cfg_path")
    stream_name_str = cfg.get("stream_name")
    pipeline = Pipeline(pipeline_cfg_file=pipeline_cfg_file_path,
                        stream_name=stream_name_str,
                        parser=ssd_mobilenet_fpn_parsing_func,
                        shared_params=shared_params)

    ssd_inference = InferenceExecutor([pipeline, transform_coco_cat_id],
                                      data_loader=coco_data_loader,
                                      name="ssd_mobilenet_fpn_inference",
                                      shared_params=shared_params,
                                      verbose=False)
    coco_annotation_path = cfg.get("label_loading_path")
    criterion_coco = COCOEvaluation(loading_path=coco_annotation_path)
    ssd_evaluator = Evaluator(criterion_coco, inference_executor=ssd_inference)
    eval_score = ssd_evaluator.eval()
    print(f"Eval score is {eval_score}.")
