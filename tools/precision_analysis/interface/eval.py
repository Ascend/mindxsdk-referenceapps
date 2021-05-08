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

from executor.inference import InferenceExecutor
from indicator.criterion import Criterion


class Evaluator(object):
    def __init__(self,
                 criterion,
                 inference_executor=None,
                 inference_result=None):
        if not isinstance(criterion, (list, tuple)):
            criterion = [criterion]

        self.inference_executor = inference_executor
        self.inference_result = inference_result
        self.criterion = criterion

        for ctn in self.criterion:
            if not isinstance(ctn, Criterion):
                raise ValueError("Given criterion must be an instance of the "
                                 "class of Criterion in package "
                                 "indicator.criterion.")

        if not (inference_executor or inference_result):
            raise IOError("Your must assign one inference method.")

        if inference_executor and inference_result:
            raise IOError("Only one inference method can be assigned.")

        if not isinstance(inference_executor, InferenceExecutor):
            raise ValueError("Param inference_executor must be the instance "
                             "of InferenceExecutor in package "
                             "executor.inference")

    def eval(self):
        if self.inference_executor:
            ret = self.inference_executor.execute()

        else:
            ret = self.inference_result

        eval_score = {}
        for ctn in self.criterion:
            score = ctn(ret)
            ctn_name = type(ctn).__name__
            eval_score[ctn_name] = score

            print(f"The score of {ctn_name} is {score} .")

        return eval_score
