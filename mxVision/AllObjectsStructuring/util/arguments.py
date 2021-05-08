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

import argparse


def get_args_for_all_object_structurization():
    """
    module for argument configuration
    """
    parser = argparse.ArgumentParser('Parameters for all object '
                                     'structurization.')
    parser.add_argument('-face-feature-pipeline-name',
                        type=str,
                        default='face-feature',
                        help='indicate the name of face feature pipeline path',
                        dest='face_feature_pipeline_name')
    parser.add_argument('-face-feature-pipeline-path',
                        type=str,
                        default='./pipeline/face_registry.pipeline',
                        help='indicate the path of face feature pipeline path',
                        dest='face_feature_pipeline_path')
    parser.add_argument('-face-root-path',
                        type=str,
                        default="./faces_to_register",
                        help='indicate the root path of face images.',
                        dest='face_root_path')
    parser.add_argument('-canvas-size',
                        type=int,
                        nargs=2,
                        default=[720, 1280],
                        help='indicate the width and height of canvas which '
                             'to place the cropped faces.',
                        dest='canvas_size')
    parser.add_argument('-index-loading-path_large',
                        type=str,
                        default="./database/large_base.index",
                        help='indicate the loading path of the large index.',
                        dest='index_loading_path_large')
    parser.add_argument('-index-loading-path-little',
                        type=str,
                        default="./database/little_base.index",
                        help='indicate the root path of the little index.',
                        dest='index_loading_path_little')
    parser.add_argument('-index-vector-dimension',
                        type=int,
                        default=256,
                        help='specify the dimension of face feature vector',
                        dest='index_vector_dimension')
    parser.add_argument('-index-base-size',
                        type=int,
                        default=400000,
                        help='specify the dimension of initialized training '
                             'matrix for the large base.',
                        dest='index_base_size')
    parser.add_argument('-index-cluster-count',
                        type=int,
                        default=8192,
                        help='specify the cluster number for ivf.',
                        dest='index_cluster_count')
    parser.add_argument('-index-topk',
                        type=int,
                        default=1,
                        help='specify the number of nearest points by sort.',
                        dest='index_topk')
    parser.add_argument('-index-little-device-ids',
                        type=int,
                        nargs="+",
                        default=[2],
                        help='specify the device assignment for little index.',
                        dest='index_little_device_ids')
    parser.add_argument('-index-large-device-ids',
                        type=int,
                        nargs="+",
                        default=[3],
                        help='specify the device assignment for large index.',
                        dest='index_large_device_ids')
    parser.add_argument('-idx2face-name-map-path',
                        type=str,
                        default='./database/idx2face.json',
                        help='indicate the path of idx to face name mapping '
                             'file.',
                        dest='idx2face_name_map_path')
    parser.add_argument('-main-pipeline-only',
                        type=bool,
                        default=False,
                        help='whether only run main pipeline.',
                        dest='main_pipeline_only')
    parser.add_argument('-main-pipeline-name',
                        type=str,
                        default='detection',
                        help='name of all object structurization pipeline.',
                        dest='main_pipeline_name')
    parser.add_argument('-main-pipeline-path',
                        type=str,
                        default='./pipeline/AllObjectsStructuring.pipeline',
                        help='path of all object structurization pipeline.',
                        dest='main_pipeline_path')
    parser.add_argument('-main-pipeline-channel-count',
                        type=int,
                        default=12,
                        help='channle count for given pipeline.',
                        dest='main_pipeline_channel_count')
    parser.add_argument('-main-keys2fetch',
                        type=str,
                        nargs="+",
                        default=[
                            'face_attribute', 'face_feature',
                            'mxpi_parallel2serial2', 'motor_attr',
                            'car_plate', 'ReservedFrameInfo',
                            'pedestrian_attribute', 'pedestrian_reid',
                            'vision', 'object'
                        ],
                        help='specify the keys to fetch their corresponding '
                             'proto buf.',
                        dest='main_keys2fetch')
    parser.add_argument('-main-stream-bbox-keys2fetch',
                        type=str,
                        nargs="+",
                        default=['mxpi_framealign0'],
                        help='specify the keys to fetch stream frame and mot data  ',
                        dest='main_stream_bbox_keys2fetch'
                        )
    parser.add_argument('-main-save-fig',
                        type=bool,
                        default=False,
                        help='specify whether to save detected object as '
                             'image.',
                        dest='main_save_fig')
    parser.add_argument('-main-base64-encode',
                        type=bool,
                        default=True,
                        help='specify whether to encode detected object into '
                             'base64.',
                        dest='main_base64_encode')
    parser.add_argument('-display-stream-bbox-data',
                        type=bool,
                        default=False,
                        help='specify whether to display stream and bbox data ',
                        dest='display_stream_bbox_data')
    parser.add_argument('-web-display-ip',
                        type=str,
                        default='0.0.0.0',
                        help='the ip of WebSocket server',
                        dest='web_display_ip')
    parser.add_argument('-web-display-port',
                        type=int,
                        default=30020,
                        help='the port of WebSocket server',
                        dest='web_display_port')

    args = parser.parse_args()

    return args
