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
import cv2

from executor.data.dataloader import DataLoader


class ImageLoaderDir(DataLoader):
    def load_dataset(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
            image_format    indicate 'RGB' or 'BGR'
            suffix_set      such as .jpg .json ...
            shared_params   storing the params such as image height, weight ...
        :return:
        """
        image_format = kwargs.get("image_format")
        self.image_format = image_format if image_format else "BGR"
        if self.image_format not in ["BGR", "RGB"]:
            raise ValueError("Please specify the image format within ['BGR', "
                             "'RGB']")

        suffix_set = kwargs.get("suffix_set")
        self.suffix_set = suffix_set if suffix_set else (".jpg", )
        if not isinstance(self.suffix_set, (list, tuple)):
            raise ValueError("You should assign one or one set of suffixes.")

        for item in self.suffix_set:
            if not isinstance(item, str):
                raise ValueError("Suffix format looks like '.jpg', '.json'.")

        shared_params = kwargs.get("shared_params")
        self.shared_params = shared_params if shared_params is not None \
            else {}

        self.file_list = os.listdir(self.loading_path)
        self.sample_cnt = len(self.file_list)

    def __iter__(self):
        idx = 0
        while idx < self.sample_cnt:
            file = self.file_list[idx]
            print(f"filename: {file}")
            _, suffix = os.path.splitext(file)
            if self.suffix_set and suffix not in self.suffix_set:
                idx += 1
                continue
            img_path = os.path.join(self.loading_path, file)
            image = cv2.imread(img_path)
            if self.image_format == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w, _ = image.shape
            self.shared_params["height"] = h
            self.shared_params["weight"] = w
            self.shared_params["file_name"] = file

            yield image
            idx += 1
