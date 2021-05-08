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

import json
import os
import cv2
import numpy as np
from PIL import Image

from multiprocessing import Lock

from util.checker import check_loading_path, check_saving_path


class Canvas:
    def __init__(self, desired_height, desired_width, resized_height=200):
        self.canvas_height = desired_height
        self.canvas_width = desired_width
        self.resized_height = resized_height

    @staticmethod
    def read(path):
        if not isinstance(path, str):
            raise ValueError("Please assign one string to indicates the  "
                             "image path.")

        if not os.path.exists(path):
            raise ValueError("Image directory does not exist.")

        return cv2.imread(path)

    def deploy(self, image):
        canvas = np.zeros([self.canvas_height, self.canvas_width, 3],
                          dtype=np.uint8)
        h, w, _ = image.shape
        canvas[0:h, 0:w, :] = image
        return canvas

    @staticmethod
    def encode(image):
        _, buf = cv2.imencode(".jpg", image)
        return Image.fromarray(np.uint8(buf)).tobytes()

    def resize(self, image):
        h, w, _ = image.shape
        resized_width = round(self.resized_height / h * w)
        return cv2.resize(image, (resized_width, self.resized_height))

    def __call__(self, path, binary=True):
        image = self.read(path)
        image = self.resize(image)
        enlarged_img = self.deploy(image)

        return self.encode(enlarged_img) if binary else enlarged_img


class Collector:
    def __init__(self, root_dir, object_name, suffix=(".jpg",)):
        if not isinstance(root_dir, str):
            raise ValueError("Please assign one string to indicates the  "
                             "root path which containing images.")

        if not os.path.exists(root_dir):
            raise ValueError("Root directory does not exist.")

        self.root_dir = root_dir
        self.object = object_name
        self.suffix = suffix
        self.collection = {}
        self.collect(self.root_dir, self.object)

    def collect(self, current_path, current_name):
        collection = os.listdir(current_path)
        for child in collection:
            child_path = os.path.join(current_path, child)
            name, tail = os.path.splitext(child)
            child_name = current_name + "_" + name
            if os.path.isdir(child_path):
                self.collect(child_path, child_name)

            elif tail in self.suffix:
                self.collection[child_name] = child_path

    def __getitem__(self, item):
        return self.collection.get(item)

    def __iter__(self):
        for key, item in self.collection.items():
            yield key, item


class Map:
    def __init__(self, map_path=None, multi_flag=False):
        self.map = {}
        self.multi_flag = multi_flag
        self.lock = Lock()
        if map_path:
            check_loading_path(map_path,
                               "loading map path",
                               suffix_set=".json")
            with open(map_path, 'r') as f:
                self.map = json.load(f)

            print(f"Your map has restored from {map_path}")

    def __getitem__(self, item):
        if item not in self.map:
            raise ValueError("Given idx not found.")

        return self.map.get(item)

    def __setitem__(self, key, value):
        if self.multi_flag:
            self.lock.acquire()
        self.map[key] = value
        if self.multi_flag:
            self.lock.release()

    def save(self, saving_path):
        check_saving_path(saving_path, "saving map path", suffix_set=".json")
        with open(saving_path, 'w') as f:
            json.dump(self.map, f)

        print(f"Your map has been saved at {saving_path}")


def parse_feature(string):
    dictionary = json.loads(string)
    if dictionary.get("MxpiFeatureVector") is not None:
        return dictionary.get("MxpiFeatureVector")[0].get("featureValues")


if __name__ == "__main__":
    path_collection = Collector("../faces_to_register", "face")
    idx2face = Map()
    my_canvas = Canvas(720, 1280)
    document = "face"
    if not os.path.exists(document):
        os.makedirs(document)
    for image_key, image_path in path_collection:
        print(f"{image_key}: {image_path}")
        idx2face[image_key] = image_path
        enlarged_image = my_canvas(image_path, binary=False)
        save_dir = os.path.join(document, f"{image_key}.jpg")
        cv2.imwrite(save_dir, enlarged_image)

    print("Done!")
