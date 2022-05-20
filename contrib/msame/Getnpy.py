# -*- coding: utf-8 -*- 
# Copyright 2020 Huawei Technologies Co., Ltd
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
# ============================================================================
import mindx.sdk as sdk
import numpy as np
image_path = "dog.jpg"

im = sdk.image(image_path, 0)
resize_img = sdk.dvpp.resize(im, height=416, width=416)
t = resize_img.get_tensor()
t.to_host()
npy = np.array(t)
np.save("dog.npy",npy)

