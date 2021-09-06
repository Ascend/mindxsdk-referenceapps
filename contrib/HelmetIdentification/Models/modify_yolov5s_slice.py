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
# Get the largest int value
INT_MAX = sys.maxsize
# Read model save path
MODEL_PATH = sys.argv[1]
# Load the MODEL
MODEL = onnx.load(MODEL_PATH)

def get_node_by_name(nodes, name):
    """
    gain node by names
    """
    for n_nodes in nodes:
        if n_nodes.name == name:
            return n_nodes
    return -1


# remove node of Slice_24
MODEL.graph.node.remove(get_node_by_name(MODEL.graph.node, "Slice_24"))
MODEL.graph.node.remove(get_node_by_name(MODEL.graph.node, "Slice_34"))

# Set the output size of Transpose after slice4 and slice24,slice9, slice19, slice29, slice39
PROB_INFO1 = onnx.helper.make_tensor_value_info('to_slice9', onnx.TensorProto.FLOAT, [1, 3, 640, 320])
PROB_INFO3 = onnx.helper.make_tensor_value_info('to_slice19', onnx.TensorProto.FLOAT, [1, 3, 640, 320])
PROB_INFO5 = onnx.helper.make_tensor_value_info('from_slice9', onnx.TensorProto.FLOAT, [1, 3, 320, 320])
PROB_INFO6 = onnx.helper.make_tensor_value_info('from_slice19', onnx.TensorProto.FLOAT, [1, 3, 320, 320])
PROB_INFO7 = onnx.helper.make_tensor_value_info('from_slice29', onnx.TensorProto.FLOAT, [1, 3, 320, 320])
PROB_INFO8 = onnx.helper.make_tensor_value_info('from_slice39', onnx.TensorProto.FLOAT, [1, 3, 320, 320])
# Transpose after slice4
NODE1 = onnx.helper.make_node(
    'Transpose',
    inputs=['171'],
    outputs=['to_slice9'],
    perm=[0, 1, 3, 2]
)
# Transpose after slice24
NODE3 = onnx.helper.make_node(
    'Transpose',
    inputs=['181'],
    outputs=['to_slice19'],
    perm=[0, 1, 3, 2]
)
# add Transpose after slice9
NODE5 = onnx.helper.make_node(
    'Transpose',
    inputs=['from_slice9'],
    outputs=['176'],
    perm=[0, 1, 3, 2]
)
# add Transpose after slice19
NODE6 = onnx.helper.make_node(
    'Transpose',
    inputs=['from_slice19'],
    outputs=['186'],
    perm=[0, 1, 3, 2]
)
# add Transpose after slice29
NODE7 = onnx.helper.make_node(
    'Transpose',
    inputs=['from_slice29'],
    outputs=['196'],
    perm=[0, 1, 3, 2]
)
# add Transpose after slice39
NODE8 = onnx.helper.make_node(
    'Transpose',
    inputs=['from_slice39'],
    outputs=['206'],
    perm=[0, 1, 3, 2]
)
# add the above node
MODEL.graph.node.append(NODE1)
MODEL.graph.node.append(NODE3)
MODEL.graph.node.append(NODE5)
MODEL.graph.node.append(NODE6)
MODEL.graph.node.append(NODE7)
MODEL.graph.node.append(NODE8)

# slice9 slice19 Change shaft
MODEL.graph.initializer.append(onnx.helper.make_tensor('starts_9', onnx.TensorProto.INT64, [1], [0]))
MODEL.graph.initializer.append(onnx.helper.make_tensor('ends_9', onnx.TensorProto.INT64, [1], [INT_MAX]))
MODEL.graph.initializer.append(onnx.helper.make_tensor('axes_9', onnx.TensorProto.INT64, [1], [2]))
MODEL.graph.initializer.append(onnx.helper.make_tensor('steps_9', onnx.TensorProto.INT64, [1], [2]))
# add Slice_9
NEWNODE1 = onnx.helper.make_node(
    'Slice',
    name='Slice_9',
    inputs=['to_slice9', 'starts_9', 'ends_9', 'axes_9', 'steps_9'],
    outputs=['from_slice9'],
)
# remove Original node Slice_9
MODEL.graph.node.remove(get_node_by_name(MODEL.graph.node, "Slice_9"))
MODEL.graph.node.insert(9, NEWNODE1)
# add Slice_19
NEWNODE2 = onnx.helper.make_node(
    'Slice',
    name='Slice_19',
    inputs=['to_slice19', 'starts_9', 'ends_9', 'axes_9', 'steps_9'],
    outputs=['from_slice19'],
)
# remove Original node Slice_19
MODEL.graph.node.remove(get_node_by_name(MODEL.graph.node, "Slice_19"))
MODEL.graph.node.insert(19, NEWNODE2)

# slice29 slice39 Change shaft
MODEL.graph.initializer.append(onnx.helper.make_tensor('starts_29', onnx.TensorProto.INT64, [1], [1]))
MODEL.graph.initializer.append(onnx.helper.make_tensor('ends_29', onnx.TensorProto.INT64, [1], [INT_MAX]))
MODEL.graph.initializer.append(onnx.helper.make_tensor('axes_29', onnx.TensorProto.INT64, [1], [2]))
MODEL.graph.initializer.append(onnx.helper.make_tensor('steps_29', onnx.TensorProto.INT64, [1], [2]))
# add Slice_29
NEWNODE3 = onnx.helper.make_node(
    'Slice',
    name='Slice_29',
    inputs=['to_slice9', 'starts_29', 'ends_29', 'axes_29', 'steps_29'],
    outputs=['from_slice29'],
)
# remove Original node Slice_29
MODEL.graph.node.remove(get_node_by_name(MODEL.graph.node, "Slice_29"))
MODEL.graph.node.insert(29, NEWNODE3)
# add Slice_39
NEWNODE4 = onnx.helper.make_node(
    'Slice',
    name='Slice_39',
    inputs=['to_slice19', 'starts_29', 'ends_29', 'axes_29', 'steps_29'],
    outputs=['from_slice39'],
)
# remove Original node Slice_39
MODEL.graph.node.remove(get_node_by_name(MODEL.graph.node, "Slice_39"))
MODEL.graph.node.insert(39, NEWNODE4)
# Save the modified onnx model
onnx.save(MODEL, sys.argv[1].split('.')[0] + "_t.onnx")
