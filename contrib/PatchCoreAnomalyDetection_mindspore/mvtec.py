# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
import json
import os
from pathlib import Path

import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter
from mindspore.dataset.vision import CenterCrop
from mindspore.dataset.vision import Resize
from mindspore.dataset.vision import ToTensor
from mindspore.dataset.transforms import TypeCast
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import Normalize
from PIL import Image
from enum import Enum

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset():
    """MVTecDataset"""

    def __init__(self, source, classname, transform, gt_transform, phase, is_json=False, split=DatasetSplit.TEST):
        root = os.path.join(source, classname)
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')

        self.is_json = is_json
        self.transform = transform
        self.gt_transform = gt_transform
        self.source = source
        self.classname = classname
        self.split = split
        self.train_val_split = 1
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        """load_dataset"""
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = sorted(glob.glob(os.path.join(self.img_path, defect_type) + "/*.png"))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = sorted(glob.glob(os.path.join(self.img_path, defect_type) + "/*.png"))
                gt_paths = sorted(glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png"))
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img = self.transform(img)

        if gt == 0:
            gt = np.zeros((1, np.array(img).shape[-2], np.array(img).shape[-2])).tolist()
        else:
            gt = Image.open(gt)
            gt = np.array(gt)
            gt = self.gt_transform(gt)

        if self.is_json:
            return os.path.basename(img_path[:-4]), img_type
        return img, img, gt, label, idx, img_path

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


def createDatasetJson(dataset_path, category, data_transforms, gt_transforms):
    """createDatasetJson"""
    train_json_path = os.path.join(dataset_path, category, '{}_{}.json'.format(category, 'train'))
    test_json_path = os.path.join(dataset_path, category, '{}_{}.json'.format(category, 'test'))

    if not os.path.isfile(train_json_path):
        print(train_json_path)
        os.mknod(train_json_path)
        train_data = MVTecDataset(root=os.path.join(dataset_path, category),
                                  transform=data_transforms, gt_transform=gt_transforms, phase='train', is_json=True)
        train_label = {}
        train_data_length = train_data.__len__()
        for i in range(train_data_length):
            single_label = {}
            name, img_type = train_data.__getitem__(i)
            single_label['name'] = name
            single_label['img_type'] = img_type
            train_label['{}'.format(i)] = single_label

        json_path = Path(train_json_path)
        with json_path.open('w') as json_file:
            json.dump(train_label, json_file)

    if not os.path.isfile(test_json_path):
        os.mknod(test_json_path)
        test_data = MVTecDataset(root=os.path.join(dataset_path, category),
                                 transform=data_transforms, gt_transform=gt_transforms, phase='test', is_json=True)
        test_label = {}
        test_data_length = test_data.__len__()
        for i in range(test_data_length):
            single_label = {}
            name, img_type = test_data.__getitem__(i)
            single_label['name'] = name
            single_label['img_type'] = img_type
            test_label['{}'.format(i)] = single_label

        json_path = Path(test_json_path)
        with json_path.open('w') as json_file:
            json.dump(test_label, json_file)

    return train_json_path, test_json_path


def createDataset(dataset_path, category, resize=256, imagesize=224):
    """createDataset"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = Compose([
        Resize((resize, resize), interpolation=Inter.AREA),
        CenterCrop(imagesize),
        ToTensor(),
        Normalize(mean=mean, std=std, is_hwc=False)
    ])
    gt_transforms = Compose([
        Resize((resize, resize), interpolation=Inter.AREA),
        CenterCrop(imagesize),
        ToTensor()
    ])

    train_json_path, test_json_path = createDatasetJson(dataset_path, category, data_transforms, gt_transforms)

    train_data = MVTecDataset(source=dataset_path, classname=category,
                              transform=data_transforms, gt_transform=gt_transforms, phase='train')
    test_data = MVTecDataset(source=dataset_path, classname=category,
                             transform=data_transforms, gt_transform=gt_transforms, phase='test')

    train_dataset = ds.GeneratorDataset(train_data,
                                        column_names=['image', 'image2', 'mask', 'is_anomaly', 'idx', 'img_path'],
                                        shuffle=False)
    test_dataset = ds.GeneratorDataset(test_data,
                                       column_names=['image', 'image2', 'mask', 'is_anomaly', 'idx', 'img_path'],
                                       shuffle=False)

    type_cast_float16_op = TypeCast(mstype.float32)
    train_dataset = train_dataset.map(operations=type_cast_float16_op, input_columns="image")
    test_dataset = test_dataset.map(operations=type_cast_float16_op, input_columns="image")

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    test_dataset = test_dataset.batch(1, drop_remainder=False)

    return train_dataset, test_dataset, train_json_path, test_json_path
