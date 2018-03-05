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

writeProposals = 1

drawGth = 0
drawResults = 0
showResults = 0

experiment_name = 'caffe-ssd-fpn-convfuse-reduce-1024'


'''
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array((104.0, 117.0, 123.0), dtype=np.float32)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
'''

import os
import os.path

def process(data, w, h, thresh):
    faceList = []
    flatten = data[0,0,:,:]
    num = flatten.shape[0]
    for j in range(num):
        rec=flatten[j,:]
        if (rec[2] < thresh) :
            break
        if (not rec[1] == 1) :
            continue
        x1=float(rec[3])*w
        y1=float(rec[4])*h
        x2=float(rec[5])*w
        y2=float(rec[6])*h
        '''
        wface = x2-x1
        hface = y2-y1
        ar=float(hface)/wface
        if x1<0 or x2<0 or y1<0 or y2<0 or ar<0.7:
            continue
        '''
        #print x1, x2, y1, y2, rec[2]
        faceList.append([x1, x2, y1, y2, rec[2]])
    return faceList

def drawFaces(img, faces, ellipses):
    font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体      
    for face in faces:
        prob = face[4]
        xmin = int(face[0])
        xmax = int(face[1])
        ymin = int(face[2])
        ymax = int(face[3])
        cv2.putText(img,str(prob),(max(int(xmin),0), max(int(ymin),0)),font,0.4,(0,0,255),1)#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        cv2.rectangle( img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0) )
    for ellipse in ellipses:      
        centerx = int( float(ellipse[3]) )
        centery = int( float(ellipse[4]) )
        major = int( float(ellipse[0]) )
        minor = int( float(ellipse[1]) )
        angle = float( ellipse[2] )
        cv2.ellipse( img, (centerx,centery), (major,minor), angle/math.pi*180, 0, 360, (255,0,0) )
    return img
    
# Read
#f_gth = open("/home/smiles/hz/databases/FDDB/FDDB-folds/FDDB-fold-all-groundtruth.txt", 'r')
f = open("/home/smiles/hz/databases/WIDER-face/annotation_2017/WIDER_val.txt", 'r')
line = f.readline().strip('\n')
# Write
'''
if writeProposals:
    ff = open("/home/smiles/hz/databases/FDDB/evaluation/proposals/"+experiment_name+".txt", 'w')
'''
import multiprocessing

def forward(img):
    net = caffe.Net('/home/smiles/hz/caffe-ssd/models/VGGNet/VOC0712/SSD_fpn/conv_fuse/reduce/deploy.prototxt',
                    '/home/smiles/hz/caffe-ssd/models/VGGNet/VOC0712/SSD_fpn/conv_fuse/reduce/VGGNet_SSD_512x512_concat_convfuse_reduce_iter_40000.caffemodel',
                    caffe.TEST)
    caffe.set_mode_gpu()
    hpic = img.shape[0]
    wpic = img.shape[1]
    #break
    if hpic <= 3072 and wpic <= 3072:
        net.blobs['data'].reshape(1,3,hpic,wpic)
    else:
        net.blobs['data'].reshape(1,3,1024,1024)
        img = cv2.resize(img, net.blobs['data'].data.shape[2:])    # resize
    
    img_ = img - [104.0, 117.0, 123.0]                         # mean
    img_ = np.transpose(img_, (2,0,1))                          # transpose
    #print img_.shape
    net.blobs['data'].data[...] = img_
    out = net.forward()
    print(img.shape)
    
while line:
    res = np.zeros((1024,1024))
    #cv2.imshow(' ', res)
    #cv2.waitKey(0)
    for i in range(1024):
        for j in range(1024):
            if i*j ==0 or i%50 != 0 or j%50 != 0:
                continue
            img = np.ones((i,j,3))

            th = multiprocessing.Process(target=forward, args=(img,))
            th.start()
            th.join()
    cv2.imshow(' ', res)
    cv2.waitKey(0)
f.close()
