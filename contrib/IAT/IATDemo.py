from mindx.sdk import base
import mindx.sdk as sdk
import cv2
import numpy as np
import os.path
model_path = "/home/nankaigcs1/IAT/models/IAT_lol.om"       #模型的路径
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
    IAT = base.model(model_path, device_id)       #创造模型对象
    image = np.array(cv2.imread(image_path))
    image = sdk.Image(image)
    image.to_device(device_id)
    print(image)
    # resize_img = sdk.dvpp.resize(im, height=600, width=400)  #对图片进行resize处理
    image = image.get_tensor()   #获取图片的Tensor对象
    outputs = IAT.infer(image)          #使用模型对Tensor对象进行推理
    cv2.imshow(outputs)

if __name__ == "__main__":
    print("hello")
    try:
        infer()
    except Exception as e:
        print(e)