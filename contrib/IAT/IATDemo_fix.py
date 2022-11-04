from base64 import decode
import numpy as np
from mindx.sdk import base
from mindx.sdk.base import Tensor, Model, log
import cv2
import os.path
model_path = "./models/IAT_lol.om"       #模型的路径
image_path = "data/test.png"             #输入图片
device_id = 0                            #芯片ID

# class SSIM(torch.nn.Module):
#     def __init__(self, channels=3):
#
#         super(SSIM, self).__init__()
#         self.win = fspecial_gauss(11, 1.5, channels)
#
#     def forward(self, X, Y, as_loss=True):
#         assert X.shape == Y.shape
#         if as_loss:
#             score = ssim(X, Y, win=self.win)
#             return 1 - score.mean()
#         else:
#             with torch.no_grad():
#                 score = ssim(X, Y, win=self.win)
#             return score
#
# class PSNR(nn.Module):
#     def __init__(self, max_val=0):
#         super().__init__()
#
#         base10 = torch.log(torch.tensor(10.0))
#         max_val = torch.tensor(max_val).float()
#
#         self.register_buffer('base10', base10)
#         self.register_buffer('max_val', 20 * torch.log(max_val) / base10)
#
#     def __call__(self, a, b):
#         mse = torch.mean((a.float() - b.float()) ** 2)
#
#         if mse == 0:
#             return 0
#
#         return 10 * torch.log10((1.0 / mse))

def infer():
    print(os.path.exists(model_path))
    IAT = Model(model_path, device_id) #创造模型对象

    # imageProcessor0 = ImageProcessor(device_id) #3.0RC2版本还未就绪
    # decodedImg = imageProcessor0.decode(image_path, base.bgr)

    # imageProcessor1 = ImageProcessor(device_id)
    # size_cof=(600,400)
    # resizeImg = imageProcessor1.resize(decodedImg, size_cof)
    image = np.array(cv2.imread(image_path))
    size_cof=(600,400)
    resizeImg = cv2.resize(image, size_cof, interpolation= cv2.INTER_LINEAR)
    imageTensor = [Tensor(resizeImg)]
    outputs = IAT.infer(imageTensor)          #使用模型对Tensor对象进行推理

    tar0 = outputs[0]
    tar0.to_host()

    res0_array = np.array(tar0)
    print(res0_array)


if __name__ == "__main__":
    print("hello")
    try:
        infer()
    except Exception as e:
        print(e)