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

from multiprocessing import Lock
from multiprocessing import Queue


class MultiprocessingQueue:
    def __init__(self):
        self.queue = Queue()
        self.lock = Lock()

    def put(self, element):
        self.lock.acquire()
        self.queue.put(element)
        self.lock.release()

    def get(self):
        return self.queue.get(True)


class MultiprocessingDictQueue:
    def __init__(self, channel_count):
        self.chl_ret = dict()
        for i in range(channel_count):
            self.chl_ret[i] = MultiprocessingQueue()

    def put(self, element, channel_id):
        self.chl_ret[int(channel_id)].put(element)

    def get(self, channel_id):
        return self.chl_ret[int(channel_id)].get()

    def __iter__(self):
        for key, item in self.chl_ret.items():
            yield key, item
