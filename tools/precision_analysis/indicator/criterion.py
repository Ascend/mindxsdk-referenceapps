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

import os
import numpy as np
import subprocess
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from collections import Iterable

from utils.checker import check_loading_path


class Criterion(object):
    def __init__(self, loading_path, *args, **kwargs):
        """
        All configurable params should be set here.
        """
        check_loading_path(loading_path, "loading_path for label")
        self.loading_path = loading_path
        self.init_criterion(*args, **kwargs)

    def init_criterion(self, *args, **kwargs):
        raise NotImplementedError("Please specify a criterion initialization "
                                  "function")

    def eval(self, pred: Iterable) -> int or float:
        raise NotImplementedError("Please the evaluation function.")

    def __call__(self, pred):
        if not isinstance(pred, Iterable):
            raise ValueError("Param pred must be an iterable object.")

        return self.eval(pred)


class COCOEvaluation(Criterion):
    def init_criterion(self, *args, **kwargs):
        self.coco_gt = COCO(self.loading_path)
        self.img_list = kwargs.get("img_list")
        if self.img_list and not isinstance(self.img_list, list):
            raise ValueError("Please specify the image list which is used "
                             "for evaluating.")

    def eval(self, pred: Iterable or str) -> int or float:

        if isinstance(pred, str):
            check_loading_path(pred, "prediction file path")
            coco_dt = self.coco_gt.loadRes(pred)

        elif isinstance(pred, np.ndarray):
            coco_dt = self.coco_gt.loadRes(pred)

        elif isinstance(pred, list):
            obj_pred = self.convert_format(pred)
            coco_dt = self.coco_gt.loadRes(obj_pred)
        else:
            raise ValueError("Wrong pred format was given.")

        E = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        if self.img_list:
            E.params.imgIds = self.img_list
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("mAP: ", E.stats[0])
        return E.stats[0]

    @staticmethod
    def convert_format(pred):
        # Todo 解析模型推理获得的结果
        ret = []
        for item in pred:
            obj = [item["image_id"]]
            obj.extend(item["bbox"])
            obj.append(item["score"])
            obj.append(item["category_id"])
            ret.append(obj)

        return np.array(ret)


class PerCharPrecision(Criterion):
    def init_criterion(self, *args, **kwargs):
        self.label_list = []
        if not os.path.isdir(self.loading_path):
            raise ValueError("Given loading_path should be a direction.")

        file_list = os.listdir(self.loading_path)
        for file in file_list:
            if file.endswith(".jpg"):
                label = file.split(".")[0].lower()
                label = label.split('_')[-1]
                self.label_list.append(label)

    def eval(self, pred: Iterable) -> int or float:
        accuracy = []

        for index, (label, prediction) in enumerate(zip(self.label_list,
                                                        pred)):
            if not isinstance(prediction, str):
                raise ValueError("Please check your predictions, "
                                 "all predictions should be string.")

            prediction = prediction.lower()
            sample_count = len(label)
            correct_count = 0
            for i, tmp in enumerate(label):
                if i >= len(prediction):
                    break
                elif tmp == prediction[i]:
                    correct_count += 1

            try:
                accuracy.append(correct_count / sample_count)
            except ZeroDivisionError:
                if len(prediction) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)

        temp = np.array(accuracy).astype(np.float32)
        avg_accuracy = np.mean(temp, axis=0)

        print('PerChar Precision is {:5f}'.format(avg_accuracy))
        return avg_accuracy


class FullSequencePrecision(Criterion):
    def init_criterion(self, *args, **kwargs):
        self.label_list = []
        if not os.path.isdir(self.loading_path):
            raise ValueError("Given loading_path should be a direction.")

        file_list = os.listdir(self.loading_path)
        for file in file_list:
            if file.endswith(".jpg"):
                label = file.split(".")[0].lower()
                label = label.split('_')[-1]
                self.label_list.append(label)

    def eval(self, pred: Iterable) -> int or float:
        try:
            correct_count = 0
            for index, (label, prediction) in enumerate(zip(
                    self.label_list, pred)):
                if not isinstance(prediction, str):
                    raise ValueError("Please check your predictions, "
                                     "all predictions should be string.")

                prediction = prediction.lower()
                if prediction == label:
                    correct_count += 1
                else:
                    print("mistake index: " + str(index))
                    print(prediction + " :: " + label)
            avg_accuracy = correct_count / len(self.label_list)
        except ZeroDivisionError:
            if not pred:
                avg_accuracy = 1
            else:
                avg_accuracy = 0
        print('Full Sequence Precision is {:5f}'.format(avg_accuracy))

        return avg_accuracy


class TextDetectEvaluation(Criterion):
    def init_criterion(self, *args, **kwargs):
        self.gt_path = self.loading_path
        self.result_path = kwargs.get("result_path")
        self.result_zip_name = kwargs.get("result_zip_name")
        self.dt_path = os.path.join(self.result_path, self.result_zip_name)
        self.input_info = {'g': self.gt_path, 's': self.dt_path, 'o': './'}

    def prepare_dt_data(self):
        zip_command = "cd {};/usr/bin/zip -qr {} res_*.txt".format(self.result_path, self.result_zip_name)
        if subprocess.call(zip_command, shell=True) == 0:
            print("Successful zip {}".format(self.result_zip_name))
        else:
            raise ValueError("zip {} failed".format(self.result_zip_name))

    def eval(self, pred: Iterable or str):
        self.prepare_dt_data()
        return True


if __name__ == "__main__":
    loading_path = "../models/COCO/instances_val2017.json"
    criterion = COCOEvaluation(loading_path=loading_path)

    print("Done !")
