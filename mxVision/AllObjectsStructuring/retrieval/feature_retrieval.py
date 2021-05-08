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

import time
import faiss
import ascendfaiss
import numpy as np
from multiprocessing import Lock

from util.checker import check_saving_path, check_loading_path


class Searcher:
    _bucket_mode_list = ["ivf", "normal"]
    _compression_mode_list = ["flat", "pq", "sq"]
    _op_name_map = {
        "normal": "",
        "ivf": "IVF",
        "flat": "Flat",
        "pq": "PQ",
        "sq": "SQ"
    }

    def __init__(self,
                 mode="ivf_sq",
                 device_name="ascend",
                 device_id=(0, ),
                 d=128,
                 metrics=faiss.METRIC_L2,
                 nlist=16384,
                 m=64,
                 nprob=1,
                 quantizer=None,
                 sub_vec_enc_len=8,
                 sq_bits=faiss.ScalarQuantizer.QT_8bit,
                 base_mtx=None,
                 multi_flag=False):
        self.mode = mode
        self.device_name = device_name
        self.device_ids = device_id
        self.devices = None
        self.d = d
        self.metrics = metrics
        self.nlist = nlist
        self.m = m
        self.nprob = nprob
        self.sq_bits = sq_bits
        self.sub_vec_enc_len = sub_vec_enc_len
        self.quantizer = quantizer
        self.base_mtx = base_mtx
        self.config = None
        self.bucket_mode = None
        self.compression_mode = None
        self.index = None
        self.multi_flag = multi_flag
        self.lock = Lock() if multi_flag else None

        self.parse_mode()
        self.make_config()
        self.build_indexer()

    def make_config(self):
        if self.device_name == "ascend":
            self.devices = ascendfaiss.IntVector()
            self.device_ids = self.device_ids if isinstance(
                self.device_ids, (list, tuple)) else [self.device_ids]
            for device_id in self.device_ids:
                if not isinstance(device_id, int):
                    raise TypeError("Wrong device id type.")
                self.devices.push_back(device_id)
            self.config = self.get_method("config")(self.devices)

    def parse_mode(self):
        if not isinstance(self.mode, str):
            raise TypeError("Mode should be a string.")

        parsed_mode = self.mode.split("_")
        if len(parsed_mode) != 2:
            raise ValueError("mode format should like \"<bucket mode>_"
                             "<compression mode>\".")

        self.bucket_mode, self.compression_mode = parsed_mode
        if self.bucket_mode not in self._bucket_mode_list or \
                self.compression_mode not in self._compression_mode_list:
            raise ValueError(f"Note that bucket_mode should be included in"
                             f" {self._bucket_mode_list} and "
                             f"compression_mode should be included in"
                             f" {self._compression_mode_list}")

    def get_method(self, category):
        method_name = "Index" + self._op_name_map[self.bucket_mode] + \
                      self._op_name_map[self.compression_mode]

        method_name = "Ascend" + method_name if self.device_name == "ascend" \
            else method_name
        pkg = ascendfaiss if self.device_name == "ascend" else faiss

        if category == "builder":
            return getattr(pkg, method_name)
        elif category == "config":
            return getattr(pkg, method_name + "Config")
        else:
            raise ValueError("Invalid method type.")

    def get_parameters(self):
        param_head = [self.d]
        param_tail = [self.config] if self.config else []

        if self.compression_mode == "sq":
            param_tail = [self.metrics, True] + param_tail
        elif self.mode == "normal_flat":
            param_tail = [self.metrics] + param_tail

        if self.bucket_mode == "ivf":
            param_head.append(self.nlist)
            if self.device_name != "ascend":
                param_head = [self.quantizer] + param_head
                if self.quantizer is None:
                    raise ValueError("Please assign a quantizer for the "
                                     "index.")

        if self.compression_mode == "sq":
            param_head.append(self.sq_bits)
        elif self.compression_mode == "pq":
            param_head.extend([self.m, self.sub_vec_enc_len])

        return param_head + param_tail

    def build_indexer(self):
        building_method = self.get_method("builder")
        parameters = self.get_parameters()
        self.index = building_method(*parameters)

        if self.base_mtx is not None and not self.index.is_trained:
            self.train(self.base_mtx)

        if self.base_mtx is not None:
            self.add(self.base_mtx)

    def train(self, base_mtx):
        if self.multi_flag:
            self.lock.acquire()
        if self.index.is_trained:
            raise AssertionError("Index has been trained once.")

        self.index.train(base_mtx)
        if self.multi_flag:
            self.lock.release()

    def add(self, new_data, idx=None):
        if idx:
            if isinstance(idx, int):
                idx = [idx]

            if isinstance(idx, (list, tuple)):
                idx = np.array(idx, np.int)

            if not isinstance(idx, np.ndarray) or idx.dtype != np.int:
                raise TypeError("idx type is invalid.")

            if idx.shape != (new_data.shape[0], ):
                raise ValueError("idx rank mismatch.")

            if self.multi_flag:
                self.lock.acquire()
            self.index.add_with_ids(new_data, idx)
            if self.multi_flag:
                self.lock.release()

        else:
            if self.multi_flag:
                self.lock.acquire()
            idx = self.index.ntotal
            self.index.add(new_data)
            if self.multi_flag:
                self.lock.release()

        return idx

    def search(self, query, k=10):
        if self.multi_flag:
            self.lock.acquire()
        ret = self.index.search(query, k)
        if self.multi_flag:
            self.lock.release()

        return ret

    def count(self):
        return self.index.ntotal

    def set_multi_flag(self):
        self.multi_flag = True
        self.lock = Lock()

    def unset_multi_flag(self):
        self.multi_flag = False
        self.lock = None

    def save(self, saving_path):
        check_saving_path(saving_path, "saving path", suffix_set=".index")

        if self.device_name == "ascend":
            self.index = ascendfaiss.index_ascend_to_cpu(self.index)

        faiss.write_index(self.index, saving_path)
        print(f"Your index has been saved at {saving_path}")

    def load(self, loading_path):
        check_loading_path(loading_path, "loading path", suffix_set=".index")

        self.index = faiss.read_index(loading_path)
        if self.device_name == "ascend":
            self.index = ascendfaiss.index_cpu_to_ascend(
                self.devices, self.index)

        print(f"Your index has restored from {loading_path}")


def main():
    d = 128  # vector dims
    nb = 100000  # databse size
    nq = 10  # query size
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    n_new_count = 50
    x_new = np.random.random((n_new_count, d)).astype('float32')
    xq = xb[:nq, :]

    nlist = 16384
    k = 5

    quantizer = faiss.IndexFlatL2(d)

    timer = time.time()
    index = Searcher(
        mode="normal_flat",
        device_id=[10],
        quantizer=quantizer,
        d=d,
        nlist=nlist,
        base_mtx=xb)
    span = time.time() - timer
    print(f"Building a new index costs {span} s")

    timer = time.time()
    distance, indexes = index.search(xq, k)
    span = time.time() - timer
    print(f"Searching xq costs {span} s")
    print(distance)
    print("---------------------------------------------------------------")
    print(indexes)

    timer = time.time()
    index.add(x_new)
    span = time.time() - timer
    print(f"Adding x_new costs {span} s")

    print("Done!")

    index.save("test.index")

    index_new = Searcher(device_name="ascend")
    index_new.load("test.index")


if __name__ == "__main__":
    main()
