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

from utils.collection import Collection
from utils import constants

CRNN_TEST_COLLECTION = Collection("crnn_test")
SSD_MOBILENET_FPN_TEST_COLLECTION = Collection("ssd_mobilenet_fpn_test")

TEST_MAP = {constants.ModelName.CRNN.value: CRNN_TEST_COLLECTION,
            constants.ModelName.SMF.value: SSD_MOBILENET_FPN_TEST_COLLECTION}
