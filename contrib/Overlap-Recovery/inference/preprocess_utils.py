# -*- coding: utf-8 -*-

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# code reference mmcv and mmdet
import collections
import warnings
import os.path as osp

import numpy as np
import mmcv
from mmcv.utils import Registry, build_from_cfg

DATA_PIPELINES = Registry('pipeline')


@DATA_PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.
    """

    def __init__(self, to_float32=False, color_type='color',
                 channel_order='bgr', file_client_args=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        file_client_args = file_client_args or dict(backend='disk')
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, tmp_results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if tmp_results['img_prefix'] is not None:
            filename = osp.join(tmp_results['img_prefix'],
                                tmp_results['img_info']['filename'])
        else:
            filename = tmp_results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        tmp_results['filename'] = filename
        tmp_results['ori_filename'] = tmp_results['img_info']['filename']
        tmp_results['img'] = img
        tmp_results['img_shape'] = img.shape
        tmp_results['ori_shape'] = img.shape
        tmp_results['img_fields'] = ['img']
        return tmp_results


@DATA_PIPELINES.register_module()
class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, DATA_PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


@DATA_PIPELINES.register_module()
class Resize:
    """Resize images & bbox & mask.
    """

    def __init__(self, img_scale=None, multiscale_mode='range', ratio_range=None,
                 keep_ratio=True):
        bbox_clip_border = True
        backend = 'cv2'
        interpolation = 'bilinear'
        override = False
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, tmp_results):
        if 'scale' not in tmp_results:
            if 'scale_factor' in tmp_results:
                img_shape = tmp_results['img'].shape[:2]
                scale_factor = tmp_results['scale_factor']
                assert isinstance(scale_factor, float)
                tmp_results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(tmp_results)
        else:
            if not self.override:
                assert 'scale_factor' not in tmp_results, (
                    'scale and scale_factor cannot be both set.')
            else:
                tmp_results.pop('scale')
                if 'scale_factor' in tmp_results:
                    tmp_results.pop('scale_factor')
                self._random_scale(tmp_results)

        self._resize_img(tmp_results)
        self._resize_bboxes(tmp_results)
        self._resize_masks(tmp_results)
        self._resize_seg(tmp_results)
        return tmp_results

    @staticmethod
    def random_select(tmp_img_scales):
        assert mmcv.is_list_of(tmp_img_scales, tuple)
        tmp_scale_idx = np.random.randint(len(tmp_img_scales))
        tmp_img_scale = tmp_img_scales[tmp_scale_idx]
        return tmp_img_scale, tmp_scale_idx

    @staticmethod
    def random_sample(tmp_img_scales):
        assert mmcv.is_list_of(tmp_img_scales, tuple) and len(tmp_img_scales) == 2
        tmp_img_scale_long = [max(s) for s in tmp_img_scales]
        tmp_img_scale_short = [min(s) for s in tmp_img_scales]
        long_edge = np.random.randint(
            min(tmp_img_scale_long),
            max(tmp_img_scale_long) + 1)
        short_edge = np.random.randint(
            min(tmp_img_scale_short),
            max(tmp_img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(tmp_img_scale, tmp_ratio_range):
        assert isinstance(tmp_img_scale, tuple) and len(tmp_img_scale) == 2
        tmp_min_ratio, tmp_max_ratio = tmp_ratio_range
        assert tmp_min_ratio <= tmp_max_ratio
        ratio = np.random.random_sample() * (tmp_max_ratio - tmp_min_ratio) + tmp_min_ratio
        tmp_scale = int(tmp_img_scale[0] * ratio), int(tmp_img_scale[1] * ratio)
        return tmp_scale, None

    def _random_scale(self, tmp_results):
        if self.ratio_range is not None:
            tmp_scale, tmp_scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            tmp_scale, tmp_scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            tmp_scale, tmp_scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            tmp_scale, tmp_scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        tmp_results['scale'] = tmp_scale
        tmp_results['scale_idx'] = tmp_scale_idx

    def _resize_img(self, tmp_results):
        """Resize images with ``results['scale']``."""
        for tmp_key in tmp_results.get('img_fields', ['img']):
            if self.keep_ratio:
                img_resized, tmp_scale_factor = mmcv.imrescale(
                    tmp_results[tmp_key],
                    tmp_results['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img_resized.shape[:2]
                h, w = tmp_results[tmp_key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img_resized, w_scale, h_scale = mmcv.imresize(
                    tmp_results[tmp_key],
                    tmp_results['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
            tmp_results[tmp_key] = img_resized

            tmp_scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                        dtype=np.float32)
            tmp_results['img_shape'] = img_resized.shape
            # in case that there is no padding
            tmp_results['pad_shape'] = img_resized.shape
            tmp_results['scale_factor'] = tmp_scale_factor
            tmp_results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, tmp_results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in tmp_results.get('bbox_fields', []):
            bboxes = tmp_results[key] * tmp_results['scale_factor']
            if self.bbox_clip_border:
                img_shape = tmp_results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            tmp_results[key] = bboxes

    def _resize_masks(self, tmp_results):
        """Resize masks with ``results['scale']``"""
        for key in tmp_results.get('mask_fields', []):
            if tmp_results[key] is None:
                continue
            if self.keep_ratio:
                tmp_results[key] = tmp_results[key].rescale(tmp_results['scale'])
            else:
                tmp_results[key] = tmp_results[key].resize(tmp_results['img_shape'][:2])

    def _resize_seg(self, tmp_results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in tmp_results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    tmp_results[key],
                    tmp_results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    tmp_results[key],
                    tmp_results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            tmp_results[key] = gt_seg


@DATA_PIPELINES.register_module()
class Pad:
    def __init__(self, size=None, size_divisor=None, pad_to_square=False, pad_val=None):
        self.size = size
        self.size_divisor = size_divisor
        pad_val = pad_val or dict(img=0, masks=0, seg=255)
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2], pad_val=pad_val)


@DATA_PIPELINES.register_module()
class Normalize:
    """Normalize the image.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, tmp_results):
        """Call function to normalize images.
        """
        for tmp_key in tmp_results.get('img_fields', ['img']):
            tmp_results[tmp_key] = mmcv.imnormalize(tmp_results[tmp_key], self.mean, self.std,
                                                    self.to_rgb)
        tmp_results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return tmp_results


@DATA_PIPELINES.register_module()
class HWCToCHW:
    """Convert image to :obj:`Tensor` by given keys.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, tmp_results):
        for tmp_key in self.keys:
            tmp_img = tmp_results[tmp_key]
            if len(tmp_img.shape) < 3:
                tmp_img = np.expand_dims(tmp_img, -1)
            tmp_img = tmp_img.transpose(2, 0, 1)  # HWC-> CHW
            tmp_img = np.ascontiguousarray(tmp_img)
            tmp_results[tmp_key] = tmp_img
        return tmp_results


@DATA_PIPELINES.register_module()
class Collect:
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor',
                            'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        tmp_img_meta = {}
        for tmp_key in self.meta_keys:
            tmp_img_meta[tmp_key] = results[tmp_key]
        data['img_metas'] = tmp_img_meta
        for tmp_key in self.keys:
            data[tmp_key] = results[tmp_key]
        return data


def build_processor(test_pipelines):
    return Compose(test_pipelines)
