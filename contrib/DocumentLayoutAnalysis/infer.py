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

import os
import mindx.sdk as sdk
import numpy as np
from utils import generate_scale, get_result, postprocess, visualize, resize_image, decode_image

INPUT_PATH = 'input'
MODEL_PATH = 'model/layout.om'
OUTPUT_DIR = 'output'


def infer(image_path):
    device_id = 0
    m = sdk.model(MODEL_PATH, device_id)
    # Preprocess
    im = decode_image(image_path)
    arr = resize_image(im)
    image = np.expand_dims(arr, axis=0)
    # Gets the image resize ratio
    scale = generate_scale(im)
    t = sdk.Tensor(image)
    t.to_device(0)
    # Make inferences
    outputs = m.infer(t)
    # Postprocess
    out = get_result(outputs)
    result = postprocess(scale, out)
    labels = ['Text', 'Title', 'Figure', 'Figure caption', 'Table',
              'Table caption', 'Header', 'Footer', 'Reference', 'Equation']
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    visualize(image_path, result, labels, OUTPUT_DIR)


if __name__ == '__main__':
    Filelist = []
    SIZE = 0
    assert os.path.isdir(INPUT_PATH), "This input folder does not exist, pleace create it"
    for home, dirs, files in os.walk(INPUT_PATH):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    for file in Filelist:
        SIZE = SIZE + 1
        image_type = os.path.splitext(file)[-1][1:].lower()
        assert image_type == 'png' or image_type == 'jpg' or image_type == 'jpeg', \
            "The input image format is jpeg or png and jpg."
        infer(file)
    assert SIZE > 0, "image file is empty"
