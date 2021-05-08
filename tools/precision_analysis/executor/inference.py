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

from collections import Iterable


class InferenceExecutor(object):
    def __init__(self,
                 elements,
                 data_loader=None,
                 name="my_inference",
                 shared_params=None,
                 verbose=False):
        """

        :param elements: callable component of inference process
        :param data_loader:
        :param name: name of this instance
        :param shared_params: used for passing params throughout the whole
        process
        :param verbose: indicate whether print progress of inference
        """
        if isinstance(name, str):
            self.name = name
        else:
            raise ValueError("Param name should be a string.")

        # shared_params is used for transit throughout the process of
        # inference.
        if shared_params is None:
            self.shared_params = {}
        elif isinstance(shared_params, dict):
            self.shared_params = shared_params
        else:
            raise ValueError("Param params should be a string.")

        if isinstance(elements, list):
            self.pipe = elements

        else:
            self.pipe = [elements]

        for element in self.pipe:
            if not hasattr(element, '__call__'):
                raise ValueError("Param elements should be a callable "
                                 "instance or a list of callable elements.")

        self.verbose = bool(verbose)

        self.set_data_loader(data_loader)

        self.ret = None
        self.ret_collection = []

    def infer(self, input_numpy):
        self.ret = input_numpy
        for element in self.pipe:
            self.ret = element(self.ret)

        if self.verbose:
            print(f"Inference result: {self.ret}")

        return self.ret

    def __call__(self, input_numpy):
        return self.infer(input_numpy)

    def execute(self):
        if not self.data_loader:
            raise RuntimeError("You have not specify a data loader, "
                               "executing is forbidden.")

        for data in self.data_loader:
            self.infer(data)
            self.collect_ret()

        return self.ret_collection

    def push(self, element):
        self.check_element(element)
        self.pipe.append(element)

    def pop(self):
        self.pipe.pop()

    def lpush(self, element):
        self.check_element(element)
        self.pipe.index(0, element)

    def lpop(self):
        self.pipe.pop(0)

    @staticmethod
    def check_element(element):
        if hasattr(element, '__call__'):
            raise ValueError("Param element should be a callable.")

    def set_data_loader(self, data_loader):
        if data_loader and not isinstance(data_loader, Iterable):
            raise ValueError("Data_loader should be iterable.")
        else:
            self.data_loader = data_loader

    def collect_ret(self):
        if isinstance(self.ret, list):
            self.ret_collection.extend(self.ret)

        else:
            self.ret_collection.append(self.ret)


if __name__ == "__main__":
    pipeline = None
    input_numpy = None
    inference = InferenceExecutor(pipeline)
    ret = inference(input_numpy)
