# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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
import os
import time
import utils

from mindspore import nn, ops, Tensor, context
from mindspore.dataset import transforms
from vgg import Vgg16
from ss_layer import SSlayer

import numpy as np
import mindspore as ms
import mindspore.dataset.vision as vision


def artcoder(style_img_path, content_img_path, code_path, output_dir, learning_rate=0.01, 
            content_weight=1e8, style_weight=1e7, code_weight=1e15, module_size=16, 
            module_num=37, epochs=100, dis_b=80, dis_w=180, correct_b=50, correct_w=200, 
            use_activation_mechanism=1):


    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

    os.makedirs(output_dir, exist_ok=True)
    utils.del_file(output_dir)
    image_size = module_size * module_num

    transform = transforms.Compose([
        vision.Resize(image_size),
        vision.ToTensor(),
    ])

    start = time.time()
    vgg = Vgg16(requires_grad=False)
    ss_layer = SSlayer(requires_grad=False)
    end = time.time()
    print("Model loading time: {} s".format(end - start))

    start = time.time()
    style_img = utils.load_image(filename=style_img_path, size=image_size)
    content_img = utils.load_image(filename=content_img_path, size=image_size)
    code_img = utils.load_image(filename=code_path, size=image_size)
    init_img = utils.add_pattern(style_img, code_img)

    init_img = transform(init_img)[0]
    style_img = transform(style_img)[0]
    content_img = transform(content_img)[0]

    init_img = Tensor.from_numpy(np.tile(init_img, (1, 1, 1, 1)))
    style_img = Tensor.from_numpy(np.tile(style_img, (1, 1, 1, 1)))
    content_img = Tensor.from_numpy(np.tile(content_img, (1, 1, 1, 1)))

    features_style = vgg(style_img)
    features_content = vgg(content_img)

    # y is the target output. Optimized start from the content image.
    y = ms.Parameter(init_img, name='y', requires_grad=True)

    # set loss function
    mse_loss = nn.MSELoss()

    # let optimizer to optimize the tensor y
    optimizer = nn.Adam([y], learning_rate=learning_rate)

    error_matrix, ideal_result = utils.get_action_matrix(
        img_target=utils.tensor_to_pil(y),
        img_code=code_img,
        dis_b=dis_b, dis_w=dis_w
    )
    code_target = ss_layer(utils.get_target(ideal_result, b_robust=correct_b, w_robust=correct_w))

    def forward_fn(error_matrix):
        z = ops.clip_by_value(y, 0, 1)
        features_z = vgg(z)

        fz = features_z[2]
        fc = features_content[2]

        style_loss = 0
        for i in [0, 1, 2, 3]:
            # style loss
            style_loss += mse_loss(features_z[i], features_style[i])
        style_loss = style_weight * style_loss

        code_z = ss_layer(z)
        if use_activation_mechanism == 1:
            activate_num = error_matrix.sum()
            activate_weight = error_matrix
            code_z = code_z * activate_weight
            code_t = code_target * activate_weight
        else:
            code_z = code_z
            code_t = code_target
            activate_num = module_num * module_num

        code_loss = code_weight * mse_loss(code_t, code_z)
        content_loss = content_weight * mse_loss(fc, fz)
        total_loss = style_loss + code_loss + content_loss

        return total_loss, style_loss, code_loss, content_loss, activate_num

    grad_fn = ops.value_and_grad(forward_fn, grad_position=None, 
                                weights=optimizer.parameters, has_aux=True)

    def train_step(error_matrix):
        (loss, style_loss, code_loss, content_loss, activate_num), grads = grad_fn(error_matrix)
        loss = ops.depend(loss, optimizer(grads))
        return loss, style_loss, code_loss, content_loss, activate_num
    
    opt_start = time.time()
    for epoch in range(epochs):
        error_matrix, ideal_result = utils.get_action_matrix(
            img_target=utils.tensor_to_pil(ops.clip_by_value(y, 0, 1)),
            img_code=code_img,
            dis_b=dis_b, dis_w=dis_w)
        error_matrix = Tensor.from_numpy(error_matrix.astype('float32'))

        _, style_loss, code_loss, content_loss, activate_num = train_step(error_matrix)
        out = ops.clip_by_value(y, 0, 1)
    
    opt_end = time.time()
    print("Optimize {} epoch time: {} s".format(epochs, opt_end-opt_start))

    img_name = 'epoch=' + str(epoch) + '__AMN=' + str(
            "%.0f" % activate_num) + '__Lstyle=' + str(
            "%.1e" % style_loss) + '__Lcode=' + str(
            "%.1e" % code_loss) + '__Lcontent' + str(
            "%.1e" % content_loss) + '.jpg'
    utils.save_image_epoch(out, output_dir, img_name, code_img, addpattern=True)
    end = time.time()
    print("Total process time: {} s".format(end-start))
    print('Save output: ' + img_name)