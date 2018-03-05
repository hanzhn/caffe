#encoding:utf-8
import sys, os
import cv2, math
from moviepy.editor import VideoFileClip
import numpy as np

from time import time, sleep  
import mxnet as mx
import threading
from queue import Queue
#import pdb
#import traceback
from collections import namedtuple

Batch = namedtuple('Batch', ['data']) # __main__.Batch

def trans_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ = cv2.resize(img, (512,512))    # resize
    img_ = np.transpose(img_, (2,0,1))
    image = mx.nd.array(img_)
    # convert into format (batch, RGB, width, height)
    #image = mx.image.imresize(image, 512, 512)
    #image = mx.nd.transpose(image, (2,0,1))
#    image = image.astype('float32')
#    image = image - mx.nd.array([123.68,116.779,103.939]).reshape((3,1,1))
    image = image.reshape((1,3,512,512))
    return image
    
    
# mxnet init
global net
load_symbol, args, auxs = mx.model.load_checkpoint('/home/smiles/hz/mxnet-ssd/tools/caffe_converter/ssd_512_converted', 1)
symbol = mx.symbol.load('/home/smiles/hz/mxnet-ssd/tools/caffe_converter/ssd_512_converted-symbol.json')
# list the last 10 layers
all_layers = symbol.get_internals()
all_layers.list_outputs()[-1:]
fe_sym = all_layers['conv1_1_output']
fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.gpu(0), label_names=None)
fe_mod.bind(data_shapes=[('data', (1,3,512,512))])
fe_mod.set_params(args, auxs)
net = fe_mod
    
img = trans_image(cv2.imread('/home/smiles/hz/mxnet-ssd/data/demo/dog.jpg'))

fe_mod = net
fe_mod.forward(Batch([mx.nd.array(img)]))    
detections = fe_mod.get_outputs()[0].asnumpy()
print(detections)
#det = detections[0, :, :]
#data = det[np.where(det[:, 0] >= 0)[0]]

