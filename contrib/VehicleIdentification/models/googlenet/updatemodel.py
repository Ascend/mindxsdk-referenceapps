import caffe

net = caffe.Net('deploy.prototxt', 'googlenet_finetune_web_car_iter_10000.caffemodel', 'test');

net.save('googlenet.caffemodel');