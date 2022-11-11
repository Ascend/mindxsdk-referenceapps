from base64 import decode
import numpy as np
import mindx.sdk as sdk
from mindx.sdk import base
from mindx.sdk.base import Tensor, Model, Size, log, ImageProcessor
import cv2

g_model_path = "/home/nankaigcs1/IAT/models/iatsim.om"  #模型的路径
g_image_path = "/home/nankaigcs1/IAT/data/eval15/low/1.png"  #输入图片
g_result_path = "/home/nankaigcs1/IAT/data/result/"
g_dataset_dir = "/home/nankaigcs1/IAT/data/eval15/"
g_device_id = 0  #芯片ID


class SSIM():
    """
    a class to evaluate SSIM
    """
    @staticmethod
    def calcSSIM(img1, img2):
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
    def calcPSNR(a, b):
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
    image_bgr = np.array([cv2.imread(image_path)])
    image = image_bgr.transpose(0,3,1,2).astype(np.float32)/255.0
    image = np.ascontiguousarray(image,dtype=np.float32)
    return image


def infer(image_path,is_save=False):
    """
    inference a low-light image.
    :param image_path: the path of low-light image to inference
    :param is_save: whether you want to save the result as an image.
    :return: a numpy array of image
    """
    base.mx_init()
    IAT = Model(g_model_path, g_device_id) #创造模型对象

    infer_image = get_image(image_path)
    #numpy to tensor
    imageTensor = Tensor(infer_image)
    imageTensor.to_device(g_device_id)
    input = [imageTensor]

    #inference
    outputs = IAT.infer(input)          #使用模型对Tensor对象进行推理
    enhanced_img = outputs[0]
    enhanced_img.to_host()
    enhanced_img = np.array(enhanced_img)
    if is_save:
        enhanced_img = enhanced_img.reshape(3,400,600).transpose(1,2,0)*255
        cv2.imwrite(g_result_path + "result.png", enhanced_img)

    return enhanced_img


def test_precision():
    """
    evaluate precision of the model.
    :return: null
    """
    import os
    low_image_list = sorted([g_dataset_dir + "/low/" + image_name for image_name in os.listdir(g_dataset_dir + "/low/")])
    high_image_list = sorted([g_dataset_dir + "/high/" + image_name for image_name in os.listdir(g_dataset_dir + "/high/")])
    print(low_image_list)
    image_num = len(low_image_list)
    psnr_sum = 0.0
    ssim_sum = 0.0
    for i in range(image_num):
        high_image = get_image(high_image_list[i])
        enhanced_image = infer(low_image_list[i])
        psnr_sum += PSNR.calcPSNR(high_image, enhanced_image)
        ssim_sum += SSIM.calcSSIM(high_image, enhanced_image)

    psnr_avg = psnr_sum/image_num
    ssim_avg = ssim_sum/image_num
    print("PSNR: ",psnr_avg)
    print("SSIM: ",ssim_avg)
    return



    print(image_list)
if __name__ == "__main__":

    test_precision()
    # try:
    #     infer()
    # except Exception as e:
    #     print(e)

    # test_tensor()