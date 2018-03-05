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
caffe.set_mode_gpu()
net = caffe.Net('/home/smiles/hz/caffe-ssd/models/VGGNet/VOC0712/SSD_fpn/conv_fuse/reduce/deploy.prototxt',
                '/home/smiles/hz/caffe-ssd/models/VGGNet/VOC0712/SSD_fpn/conv_fuse/reduce/VGGNet_SSD_512x512_concat_convfuse_reduce_iter_40000.caffemodel',
                caffe.TEST)
# set net to batch size of 50
net.blobs['data'].reshape(1,3,640,640)
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

while line:
    print line
    img = cv2.imread(line)
    hpic = img.shape[0]
    wpic = img.shape[1]
    print(img.shape)
    #break
    if hpic <= 3072 and hpic > 256 and wpic <= 3072 and wpic >256:
        net.blobs['data'].reshape(1,3,hpic,wpic)
    #elif hpic <= 256 and wpic <= 256:
    else:
        error = open("errorPicList.txt", 'a')
        error.write(line+' '+str(hpic)+' '+str(wpic)+'\n')
        error.close()
        net.blobs['data'].reshape(1,3,1024,1024)
        img = cv2.resize(img, net.blobs['data'].data.shape[2:])    # resize
    img_ = img - [104.0, 117.0, 123.0]                         # mean
    img_ = np.transpose(img_, (2,0,1))                          # transpose
    #print img_.shape
    net.blobs['data'].data[...] = img_
    out = net.forward()
    boxes = net.blobs['detection_out'].data
    thresh = 0.1
    faces = process(boxes, wpic, hpic, thresh)

    # need draw picture
    if showResults or drawResults:
        ellipses = []
        if drawGth:
            # draw groundtruth
            name = f_gth.readline().strip('\n')
            if name != line:
                sys.exit('error!')
            num2 = int( f_gth.readline().strip('\n') )
            ellipses = []
            for i in range(num2):
                ellipses.append( f_gth.readline().strip('\n').split(' ') )
        img = drawFaces(img, faces, ellipses)
        if showResults:
            cv2.namedWindow("image")
            cv2.imshow("image", img)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
        if drawResults:
            templist = line.split('/')
            picpath = "/home/smiles/hz/databases/FDDB/"+experiment_name+"_result/"+'/'.join( templist[0:len(templist)-1] )
            if not os.path.isdir(picpath):
                #print 'New experiment. Good luck!'
                os.makedirs(picpath)
            cv2.imwrite("/home/smiles/hz/databases/FDDB/"+experiment_name+"_result/"+line+".jpg", img)
            
    if writeProposals:
        savedir = "/home/smiles/hz/databases/WIDER-face/eval_tools/proposals/"+experiment_name
        subdir = line.split('/')[-2]
        filename = line.split('/')[-1]
        if not os.path.isdir(savedir+"/"+subdir):
            os.makedirs(savedir+"/"+subdir)       
        ff = open(savedir+"/"+subdir+"/"+filename.replace('jpg','txt'), 'w')
        ff.write(line+'\n')
        ff.write(str( len(faces) )+'\n')
        for face in faces:
            prob = face[4]
            xmin = face[0]
            xmax = face[1]
            ymin = face[2]
            ymax = face[3]
            w = xmax - xmin
            h = ymax - ymin
            ff.write( "%f %f %f %f %f\n"%(xmin, ymin, w, h, prob) )
        ff.close()
    
    line = f.readline().strip('\n')
f.close()
