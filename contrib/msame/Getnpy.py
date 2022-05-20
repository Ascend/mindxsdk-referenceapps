import mindx.sdk as sdk
import numpy as np
image_path = "dog.jpg"

im = sdk.image(image_path, 0)
resize_img = sdk.dvpp.resize(im, height=416, width=416)
t = resize_img.get_tensor()
t.to_host()
npy = np.array(t)
np.save("dog.npy",npy)

