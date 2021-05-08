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


class Collection(object):
    """
    The collection use name -> class mapping, to support the customized
    implementation of a derivative class .
    To create a collection (e.g. a Metrics collection):
    Step 1:
        METRICS_COLLECTION = Collection('METRICS')
    Step 2:
        @METRICS_COLLECTION.register()
        class MyMetrics():
            ...
    """

    def __init__(self, table_name) -> None:
        """
        Args:
            table_name (str): the name of this collection
        """
        self.table_name = table_name
        self.object_map = {}

    def _do_register(self, name: str, obj: object) -> None:
        if name in self.object_map:
            raise KeyError(f"The class named '{name}' was already registered!")

        self.object_map[name] = obj

    def register(self, name=None):
        """
        Register the given class under the the name `class.__name__`.
        Please use it as a decorator.
        """
        def wrapper(cls_or_func):
            if name and isinstance(name, str):
                register_name = name
            else:
                register_name = cls_or_func.__name__

            self._do_register(register_name, cls_or_func)
            return cls_or_func

        return wrapper

    def get(self, name):
        ret = self.object_map.get(name)
        if ret is None:
            raise KeyError(f"The class named {name} does not exist!")

        return ret
