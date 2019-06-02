#coding=utf-8
#this code is used for extracting features
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
import cv2

# Make sure that caffe is on the python path:
caffe_root = '/home/smiles/hz/caffe-ssd/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
import math

writeProposals = 1

drawGth = 0
drawResults = 0
showResults = 1

import os
import os.path

def xywh2xxyy(box):
    x = int(box[0])
    y = int(box[1])
    w = int(box[2])
    h = int(box[3])
    xmin = x-w/2
    xmax = x+w/2
    ymin = y-h/2
    ymax = y+h/2
    return xmin,xmax,ymin,ymax
    
def calcIOU(box1, box2):
    '''
    calculate IoU.
    box1 and box2 is in [x,y,w,h] arrangement.
    '''
    x1 = box1[0]
    y1 = box1[1]
    w1 = box1[2]
    h1 = box1[3]
    x1min, x1max, y1min, y1max = xywh2xxyy(box1)    
    x2 = box2[0]
    y2 = box2[1]
    w2 = box2[2]
    h2 = box2[3]
    x2min, x2max, y2min, y2max = xywh2xxyy(box2)
    #print box1[:4], box2[:4]
    #print [x1min, x1max, y1min, y1max], [x2min, x2max, y2min, y2max]
    I = max(0,(min(x1max,x2max)-max(x1min,x2min))) * max(0,(min(y1max,y2max)-max(y1min,y2min)))
    U = w1*h1+w2*h2-I
    #print I, Ufloat(I/U)
    return float(I)/float(U)
    
def match(gths, steps, anchorSizes, thresh):
    '''
    get matched anchors.
    gths is in [x,y,w,h...] arrangement;
    steps is a list of steps;
    anchorSizes is a list of sizes;
    thresh is IoU threshold.
    '''
    res = []
    unmatched = []
    numUnmatchedGth = 0
    for gth in gths: 
        isUnmatched = 1
        th = thresh
        '''
        # w/h is small
        if gth[2]>0 and gth[3]>0:
            ratio = float(gth[2])/float(gth[3])
            if ratio < 0.5:
                area = abs(gth[2]*gth[3])
                gth[2] = int(area**0.5)
                gth[3] = int(area**0.5)
        '''
        # # small objects
        # if gth[2] < 10 or gth[3] < 10:    
        #     #gth[2] = max(gth[2],gth[3])
        #     #gth[3] = max(gth[2],gth[3])
        #     th = 0.1  
 
        for step, anchorSize in zip(steps, anchorSizes):
            bx = gth[0]/step
            by = gth[1]/step
            for anchorx in [bx*step-step,bx*step,bx*step+step,bx*step+step*2]:
                for anchory in [bx*step-step,by*step,by*step+step,by*step+step*2]:
                    if type(anchorSize) == list:
                        anchor = [anchorx,anchory,anchorSize[0],anchorSize[1]]
                    else:
                        anchor = [anchorx,anchory,anchorSize,anchorSize]             
                    if calcIOU(anchor,gth)>=th:
                        res.append(anchor)
                        isUnmatched = 0
        if isUnmatched:
            unmatched.append(gth)
            numUnmatchedGth += 1
    #print unmatched
    #print numUnmatchedGth
    return res, numUnmatchedGth, unmatched
    
def getGthLine(path):
    '''
    a generator to get a pic's Gth from txt file.
    path is a string of txt file path.
    '''
    '''
    a generator to get a line in txt.
    path is a string of txt file path.
    '''       
    fpic = open(path,'r')
    fgth = open('/home/smiles/hz/databases/WIDER-face/annotation_2017/wider_face_val_bbx_gt.txt','r')
    lines = fpic.readlines()
    for line in lines:
        line = line.strip('\n')
        listname = '/'.join( line.split('/')[-2:] )
        # draw groundtruth
        name = fgth.readline().strip('\n')
        if not name:
            break
        if name != listname:
            sys.exit('txt list file does not match with txt gth file!')
        num = int( fgth.readline().strip('\n') )      
        gthBoxes = []
        for i in range(num):
            box = fgth.readline().strip('\n').split(' ')
            xmin = int(box[0])
            ymin = int(box[1])
            w = int(box[2])
            h = int(box[3])
            x = xmin+w/2
            y = ymin+h/2
            gthBoxes.append( [x,y,w,h] )
        yield gthBoxes, line
    fpic.close()
    fgth.close()
   
def drawBox(img, boxes, color):
    '''
    a generator to get a pic's Gth from txt file.
    path is a string of txt file path.
    '''    
    for face in boxes:
        xmin,xmax,ymin,ymax = xywh2xxyy(face)
        cv2.rectangle( img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color )

def ratio_adjust(ratio, anchor_size):
    return [anchor_size*pow(ratio, 0.5), anchor_size/pow(ratio, 0.5)]
    
if __name__=="__main__":
    picListFile = "/home/smiles/hz/databases/WIDER-face/annotation_2017/WIDER_val.txt"
    gen = getGthLine(picListFile) 
    
    numMatchedP = 0
    numUnmatchedG = 0
    unmatchedBoxes = []
    for gth, line in gen:
        print line
        img = cv2.imread(line)
        hpic = img.shape[0]
        wpic = img.shape[1]
        thresh = 0.35
        
        #zyq settings
        #steps = [4,4,8,8,16,32,64,128]
        #anchorSizes = [6,12,24,40,80,128,256,512]
        #sfd settings
        steps = [4,8,16,32,64,128]
        anchorSizes = [16]
        extra = [32,64,128,256,512]
        # ratio: 0.76
        anchorSizes += [ratio_adjust(0.76, x) for x in extra]
        # anchorSizes += extra
        #print anchorSizes
        #exit()
        #anchorSizes = [8,[7,9],[14,18],[28,37],[56,73],[112,146],[223,294],[446,587],[543,1086]]
        #anchorSizes = [16,[28,37],[56,73],[112,146],[223,294],[446,587]]
        
        matchedPriorBox, numUnmatchedGth, unmatched = match(gth, steps, anchorSizes, thresh)
        numMatchedP += len(matchedPriorBox)
        numUnmatchedG += numUnmatchedGth
        unmatchedBoxes += unmatched
        #drawBox(img, gth, (0,255,0))
        #drawBox(img, matchedPriorBox, (255,0,0))
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        #cv2.imwrite('img.jpg',img)
    print unmatchedBoxes
    print numMatchedP, numUnmatchedG

