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


class ChannelStatus:
    """
    management for channel requiring state
    """
    def __init__(self):
        self.status_dict = {}

    def update(self, channel_id, status):
        self.status_dict[channel_id] = status

    def isAlive(self, channel_id):
        if channel_id not in self.status_dict:
            return False
        else:
            return self.status_dict[channel_id]
