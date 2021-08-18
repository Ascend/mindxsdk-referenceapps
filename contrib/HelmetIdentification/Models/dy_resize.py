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

model_path = sys.argv[1]
model = onnx.load(model_path)

def RemoveNode(graph, nodelist):
    """
    Remove Node
    """
    max_idx = len(graph.node)
    rm_cnt = 0
    for i in range(len(graph.node)):
        if i < max_idx:
            gn = graph.node[i - rm_cnt]
            if gn.name in nodelist:
                print("remove {} total {}".format(gn.name, len(graph.node)))
                graph.node.remove(gn)
                max_idx -= 1
                rm_cnt += 1


def ReplaceScales(ori_list, scales_name):
    """
    Replace Scales
    """
    n_list = []
    for j, x in enumerate(ori_list):
        if j < 2:
            n_list.append(x)
        if j == 3:
            n_list.append(scales_name)
    return n_list

# Replace Resize node
for k in range(len(model.graph.node)):
    n = model.graph.node[k]
    if n.op_type == "Resize":
        model.graph.initializer.append(
            onnx.helper.make_tensor('scales{}'.format(k), onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
        )
        newnode = onnx.helper.make_node(
            'Resize',
            name=n.name,
            inputs=ReplaceScales(n.input, 'scales{}'.format(k)),
            outputs=n.output,
            coordinate_transformation_mode='asymmetric',
            cubic_coeff_a=-0.75,
            mode='nearest',
            nearest_mode='floor'
        )
        model.graph.node.remove(model.graph.node[k])
        model.graph.node.insert(k, newnode)
        print("replace {} index {}".format(n.name, k))

node_list = ['Constant_330', 'Constant_375']
RemoveNode(model.graph, node_list)
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1].split('.')[0] + "_dbs.onnx")
