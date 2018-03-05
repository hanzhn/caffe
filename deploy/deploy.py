#coding=utf-8
#this code is used for extracting features
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# Make sure that caffe is on the python path:
caffe_root = '/home/smiles/hz/caffe-ssd/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
import cv2
sys.path.insert(0, caffe_root + 'python')
import caffe
import math

caffe.set_mode_gpu()
net = caffe.Net('/home/smiles/hz/mxnet-ssd/tools/caffe_converter/ssd512/deploy.prototxt',
                '/home/smiles/hz/mxnet-ssd/tools/caffe_converter/ssd512/VGG_VOC0712_SSD_512x512_iter_50000.caffemodel',
                caffe.TEST)
                
img = cv2.imread('/home/smiles/hz/mxnet-ssd/data/demo/dog.jpg')
img_ = cv2.resize(img, net.blobs['data'].data.shape[2:])    # resize
#img_ = img_ - [104.0, 117.0, 123.0]                         # mean
img_ = np.transpose(img_, (2,0,1))                          # transpose
print(img_)

net.blobs['data'].data[...] = img_
out = net.forward()
boxes = net.blobs["conv1_1"].data
print(boxes)
