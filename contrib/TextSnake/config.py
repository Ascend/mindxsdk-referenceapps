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

from easydict import EasyDict
import torch

config = EasyDict()

# dataloader jobs number
config.num_workers = 4

# batch_size
config.batch_size = 4

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True

config.n_disk = 15

config.output_dir = 'output/test'

config.input_size = 512

# max polygon per image
config.max_annotation = 200

# max point per polygon
config.max_points = 20

# use hard examples (annotated as '#')
config.use_hard = True

# demo tr threshold
config.tr_thresh = 0.6

# demo tcl threshold
config.tcl_thresh = 0.4

config.means = (0.485, 0.456, 0.406)
config.stds = (0.229, 0.224, 0.225)

# expand ratio in post processing
config.post_process_expand = 0.3

# merge joined text instance when predicting
config.post_process_merge = False

def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')