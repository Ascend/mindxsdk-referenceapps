import os
import cv2

FILE_PATH = './BSD68/'
for file in os.listdir(FILE_PATH):
    img_path = FILE_PATH + file
    img = cv2.imread(img_path)
    # evaluate运行前需要执行下面的resize；main运行前注释掉下面一行代码
    img = cv2.resize(img, (480, 320))
    save_path = './dataset/' + file.split('.')[0] + '.jpg'
    cv2.imwrite(save_path, img)