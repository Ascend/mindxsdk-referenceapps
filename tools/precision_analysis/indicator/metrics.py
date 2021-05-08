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

from easydict import EasyDict as edict
import numpy as np


class Metrics(object):
    def __init__(self, cfg: dict or None = None):
        """
        All configurable params should be set here.
        """
        self.multi_sample = False
        self.batch_size = None
        self.cfg = None
        self.axis = None
        self.load_cfg(cfg)

    def load_cfg(self, cfg):
        if cfg is None:
            cfg = edict()

        if not isinstance(cfg, (type(None), dict)):
            raise TypeError(f"Param 'cfg' must be None or an easydict.")

        self.cfg = cfg

        try:
            self.multi_sample = self.cfg.multi_sample
            if not isinstance(self.multi_sample, bool):
                raise TypeError(f"Param multi_sample should be a value of "
                                f"bool.")

        except AttributeError:
            pass

    def compare(self, ref: np.ndarray, tgt: np.ndarray) -> int or float:
        raise NotImplementedError

    def __call__(self, ref, tgt):
        if not isinstance(self.cfg, edict):
            raise TypeError(f"Params must be passed by a easydict instance.")

        if not isinstance(ref, type(tgt)):
            raise TypeError(f"Given ref and tgt must be the same type.")

        if isinstance(ref, list):
            ref, tgt = np.array(ref), np.array(tgt)

        if not isinstance(ref, np.ndarray):
            raise TypeError(f"Given params ref and tgt can only be a type "
                            f"within list and np.ndarray.")

        if ref.shape != tgt.shape:
            raise AssertionError(f"The shape of ref and tgt should be the "
                                 f"same.")
        self.shape = ref.shape
        self.batch_size = self.shape[0]
        if len(self.shape) == 1:
            self.multi_sample = False

        if self.multi_sample:
            ref = np.reshape(ref, (self.batch_size, -1))
            tgt = np.reshape(tgt, (self.batch_size, -1))

        self.axis = 1 if self.multi_sample else None

        return self.compare(ref, tgt)


class RE(Metrics):
    """
    relative error
    """

    def __init__(self, cfg=None):
        super(RE, self).__init__(cfg)
        try:
            self.epsilon = self.cfg.epsilon
            if not isinstance(self.epsilon, float) or self.epsilon <= 0:
                raise ValueError("Param epsilon must a float larger than 0.")
        except AttributeError:
            self.epsilon = 1e-6


class MRE(RE):
    """
    Mean relative error
    """

    def compare(self, ref, tgt):
        if self.multi_sample:
            return np.mean(np.abs(tgt - ref) / (np.abs(ref) + self.epsilon),
                           axis=1)

        else:
            return np.mean(np.abs(tgt - ref) / (np.abs(ref) + self.epsilon))


class MaxRE(RE):
    """
    Max relative error
    """

    def compare(self, ref, tgt):
        if self.multi_sample:
            return np.max(np.abs(tgt - ref) / (np.abs(ref) + self.epsilon),
                          axis=1)

        else:
            return np.max(np.abs(tgt - ref) / (np.abs(ref) + self.epsilon))


class MinRE(RE):
    """
    Min relative error
    """

    def compare(self, ref, tgt):
        if self.multi_sample:
            return np.min(np.abs(tgt - ref) / (np.abs(ref) + self.epsilon),
                          axis=1)

        else:
            return np.min(np.abs(tgt - ref) / (np.abs(ref) + self.epsilon))


class MAE(Metrics):
    """
    Mean absolute error
    """

    def compare(self, ref, tgt):
        if self.multi_sample:
            return np.mean(np.abs(tgt - ref), axis=1)

        else:
            return np.mean(np.abs(tgt - ref))


class MaxAE(Metrics):
    """
    Max absolute error
    """

    def compare(self, ref, tgt):
        if self.multi_sample:
            return np.max(np.abs(tgt - ref), axis=1)

        else:
            return np.max(np.abs(tgt - ref))


class MinAE(Metrics):
    """
    Min absolute error
    """

    def compare(self, ref, tgt):
        if self.multi_sample:
            return np.min(np.abs(tgt - ref), axis=1)

        else:
            return np.min(np.abs(tgt - ref))


class CAE(Metrics):
    """
    Cumulative absolute error
    """

    def compare(self, ref, tgt):
        if self.multi_sample:
            return np.sum(np.abs(tgt - ref), axis=1)

        else:
            return np.sum(np.abs(tgt - ref))


class MSE(Metrics):
    """
    Mean square error
    """

    def compare(self, ref, tgt):
        if self.multi_sample:
            return np.mean(np.square(tgt - ref), axis=1)

        else:
            return np.mean(np.square(tgt - ref))


class RMSE(Metrics):
    """
    Root mean square error
    """

    def compare(self, ref, tgt):
        return np.sqrt(np.mean(np.square(tgt - ref)), axis=self.axis)


class RatioAlmostEqual(Metrics):
    def __init__(self, cfg):
        super(RatioAlmostEqual).__init__(cfg)
        try:
            self.err_thresh = self.cfg.err_thresh

        except AttributeError:
            raise AttributeError(f"Please specify the threshold of error.")


class RelativeRatioAlmostEqual(RE, RatioAlmostEqual):
    def compare(self, ref, tgt):
        relative_error = np.abs(tgt - ref) / (np.abs(ref) + self.epsilon)
        element_cnt = self.shape[1]
        numerator = np.sum(relative_error > self.err_thresh, axis=self.axis)
        return numerator / element_cnt


class AbsoluteRatioAlmostEqual(RatioAlmostEqual):
    def compare(self, ref, tgt):
        absolute_error = np.abs(tgt - ref)
        element_cnt = self.shape[1]
        numerator = np.sum(absolute_error > self.err_thresh, axis=self.axis)
        return numerator / element_cnt


class CosineDistance(Metrics):
    def compare(self, ref, tgt):
        ref_norm = np.linalg.norm(ref, axis=self.axis)
        tgt_norm = np.linalg.norm(tgt, axis=self.axis)
        if self.multi_sample:
            return np.dot(ref, tgt.T).diagonal() / (tgt_norm * ref_norm)

        else:
            return np.dot(ref, tgt.T) / (tgt_norm * ref_norm)


if __name__ == "__main__":
    pass
