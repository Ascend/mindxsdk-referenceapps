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

from base64 import decode
import numpy as np
import mindx.sdk as sdk
from mindx.sdk import base
from mindx.sdk.base import Tensor, Model, Size, log, ImageProcessor
import cv2

MODEL_PATH = "/home/nankaigcs1/IAT/models/iatsim.om"   # 模型的路径
IMAGE_PATH = "/home/nankaigcs1/IAT/data/eval15/low/1.png"   # 输入图片
RESULT_PATH = "/home/nankaigcs1/IAT/data/result/"
DATASET_DIR = "/home/nankaigcs1/IAT/data/eval15/"
DEVICE_ID = 0   # 芯片ID


class SSIM():
    """
    a class to evaluate SSIM
    """
    @staticmethod
    def calc_ssim(img1, img2):
        ssim_c1 = (0.01 * 255)**2
        ssim_c2 = (0.03 * 255)**2
        img1 = img1.reshape(3, 400, 600).transpose(1, 2, 0)*255
        img2 = img2.reshape(3, 400, 600).transpose(1, 2, 0)*255
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + ssim_c1) * (2 * sigma12 + ssim_c2)) / ((mu1_sq + mu2_sq + ssim_c1) *
                                                                (sigma1_sq + sigma2_sq + ssim_c2))
        return ssim_map.mean()


class PSNR():
    """
    a class to evaluate PSNR.
    """
    @staticmethod
    def calc_psnr(a, b):
        mse = np.mean((a - b) ** 2)
        if mse == 0:
            return 0

        return 10 * np.log10((1.0 / mse))


def get_image(image_path):
    """
    get image by its path.
    :param image_path: the path of image
    :return: a numpy array of image
    """
    import os
    if not os.path.exists(image_path):
        print("image path is wrong!")
        exit(0)
    image_bgr = cv2.imread(image_path)
    imge_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    imge_rgb = cv2.resize(imge_rgb, (600, 400))
    imge_rgb = np.array([imge_rgb])
    image = imge_rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image


def infer(image_path, is_save=False):
    """
    inference a low-light image.
    :param image_path: the path of low-light image to inference
    :param is_save: whether you want to save the result as an image.
    :return: a numpy array of image
    """
    base.mx_init()


    infer_image = get_image(image_path)
    #numpy to tensor
    image_tensor = Tensor(infer_image)
    image_tensor.to_device(DEVICE_ID)
    image_tensor = [image_tensor]

    #inference
    model = Model(MODEL_PATH, DEVICE_ID)
    outputs = model.infer(image_tensor)
    enhanced_img = outputs[0]
    enhanced_img.to_host()
    enhanced_img = np.array(enhanced_img)
    if is_save:
        enhanced_img = enhanced_img.reshape(3, 400, 600).transpose(1, 2, 0) * 255
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(RESULT_PATH + image_path.split('/')[-1], enhanced_img)

    return enhanced_img


def test_precision():
    """
    evaluate precision of the model.
    :return: null
    """
    import os
    low_image_list = sorted([DATASET_DIR + "/low/" + image_name for image_name in os.listdir(DATASET_DIR + "/low/")])
    high_image_list = sorted([DATASET_DIR + "/high/" + image_name for image_name in os.listdir(DATASET_DIR + "/high/")])
    image_num = len(low_image_list)
    psnr_sum = 0.0
    ssim_sum = 0.0

    for i in range(image_num):
        high_image = get_image(high_image_list[i])
        enhanced_image = infer(low_image_list[i])
        psnr_sum += PSNR.calc_psnr(high_image, enhanced_image)
        ssim_sum += SSIM.calc_ssim(high_image, enhanced_image)

    psnr_avg = psnr_sum / image_num
    ssim_avg = ssim_sum / image_num
    print("PSNR: ", psnr_avg)
    print("SSIM: ", ssim_avg)
    return

def test_input():
    import os
    TEST_DIR = "/home/nankaigcs1/IAT/data/test/"
    SAVE_DIR = "/home/nankaigcs1/IAT/data/result/"
    low_image_list = sorted([TEST_DIR  + image_name for image_name in os.listdir(TEST_DIR)])
    image_num = len(low_image_list)

    for i in range(image_num):
        enhanced_image = infer(low_image_list[i],is_save=True)

    return

if __name__ == "__main__":
    # test_input()
    try:
        infer(IMAGE_PATH, is_save=True)
    except Exception as e:
        print(e)
