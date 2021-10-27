from PIL import Image
import sys
import os
if __name__ == '__main__':
    # test image set path
    test_image_set_path = "./Set14"
    # parse command arguments
    if len(sys.argv) == 2:
        if sys.argv[1] == '':
            print('test image set path is not valid, use default config.')
        else:
            test_image_set_path = sys.argv[1]
    # get all image files
    image_files = os.listdir(test_image_set_path)
    # convert bmp to jpg
    for test_image_path in image_files:
        image_file = test_image_set_path+"/" + test_image_path
        out_img = Image.open(image_file).convert("RGB")
        out_img.save(test_image_set_path + "-jpg/" + test_image_path[:-4] + ".jpg")
