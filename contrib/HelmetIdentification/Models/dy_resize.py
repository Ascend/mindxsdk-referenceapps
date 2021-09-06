# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import onnx

MODEL_PATH = sys.argv[1]
MODEL = onnx.load(MODEL_PATH)

def remove_node(graph, nodelist):
    """
    Remove Node
    """
    max_idx = len(graph.node)
    rm_cnt = 0
    for i in range(len(graph.node)):
        if i < max_idx:
            graph_node = graph.node[i - rm_cnt]
            if graph_node.name in nodelist:
                print("remove {} total {}".format(graph_node.name, len(graph.node)))
                graph.node.remove(graph_node)
                max_idx -= 1
                rm_cnt += 1


def replace_scales(ori_list, scales_name):
    """
    Replace Scales name:
    Leave the first two items of the input attribute of Resize unchanged
    and the third item--scales name is modified
    param:ori_list is the value of Resize.input
    """
    n_list = []
    for ori_index, x in enumerate(ori_list):
        if ori_index < 2:
            n_list.append(x)
        if ori_index == 3:
            n_list.append(scales_name)
    return n_list

# Replace Resize node
for k in range(len(MODEL.graph.node)):
    n = MODEL.graph.node[k]
    if n.op_type == "Resize":
        MODEL.graph.initializer.append(
            onnx.helper.make_tensor('scales{}'.format(k), onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
        )
        newnode = onnx.helper.make_node(
            'Resize',
            name=n.name,
            inputs=replace_scales(n.input, 'scales{}'.format(k)),
            outputs=n.output,
            coordinate_transformation_mode='asymmetric',
            cubic_coeff_a=-0.75,
            mode='nearest',
            nearest_mode='floor'
        )
        MODEL.graph.node.remove(MODEL.graph.node[k])
        MODEL.graph.node.insert(k, newnode)
        print("replace {} index {}".format(n.name, k))

NODE_LIST = ['Constant_330', 'Constant_375']
remove_node(MODEL.graph, NODE_LIST)
onnx.checker.check_model(MODEL)
onnx.save(MODEL, sys.argv[1].split('.')[0] + "_dbs.onnx")
