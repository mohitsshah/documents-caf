# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:24:23 2018

@author: genesis
"""

import sys
import parseXMLForSegmentation as pxfs
import visualizeSegments as vs
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import math

def displayLists(seg):
    if type(seg)==list:
        for s in seg:
            displayLists(s)
    else:
        cv2.imshow("Seg",seg)
        cv2.waitKey()
        
def detection(src_xml_file,src_image):
    img = cv2.imread(src_image)
    page,text_list = pxfs.fetchXMLElements(src_xml_file)
    new_img,height,width = pxfs.createNewPage(page,text_list)
    cv2.imshow("img",new_img)
#    cv2.waitKey()
    print ("HERE")
    segs = resegment(new_img)
    for s in segs:
        displayLists(s)
        
    
def gaussian(m,s,x):
    return 1/(math.sqrt(2*math.pi)*s)*math.e**(-0.5*(float(x-m)/s)**2)
def cropimg(segimg):
    xc0,yc0 = 0,0
    xc1,yc1 = segimg.shape
    for i in range(xc1):
        line = segimg[i,:]
        if np.any(line):
            xc0 = i
            break
        
    for i in range(yc1):
        line = segimg[:,i]
        if np.any(line):
            yc0 = i
            break
        
    for i in range(xc1-1,xc0,-1):
        line = segimg[i,:]
        if np.any(line):
            xc1 = i+1
            break
        
    for i in range(yc1-1,yc0,-1):
        line = segimg[:,i]
        if np.any(line):
            yc1 = i+1
            break
    
    return xc0,yc0,xc1,yc1

def resegment(segment):
    xc0,yc0,xc1,yc1 = cropimg(segment)
    csegment = segment[xc0:xc1,yc0:yc1]
    height,width = csegment.shape
    counts = []
    for i in range(height):
        line = csegment[i,:]
        if not np.any(line):
            counts.append(1)
        else:
            counts.append(0)
    counts = np.array(counts)
    partitions = np.where(counts[1:] != counts[:-1])[0] + 1
    runs = []
    start = 0
    for p in partitions:
        if counts[start]==1:
            runs.append((start,p))
        start = copy.deepcopy(p)
    
    if not np.any(np.array(runs)):
        return segment
    profile = np.zeros(height)
    for r in runs:
        profile[r[0]] = r[1]-r[0]
    normprofile = profile/np.max(profile)
    m = np.mean(normprofile)
    s = np.std(normprofile)
    
    bigGaps = np.where(normprofile>0.4)[0]
    totalGaps = np.where(normprofile>0)[0]
    print ("MEAN:",m,"STD:",s,"NO OF GAPS:",len(bigGaps))
    if len(bigGaps)<0.03*height and len(bigGaps)>0 and float(len(bigGaps))/len(totalGaps)<0.4:
        segments = []
        start = 0
        for bg in bigGaps:
            print ("GAP:",normprofile[bg],"GAUSSIAN SCORE:",1-gaussian(m,s,normprofile[bg]))
            segimg = csegment[start:bg]
            segments.append(segimg)
            start = bg
        lastsegimg = csegment[start:]
        segments.append((lastsegimg))
        returnedSegments = []
        for s in segments:
            returnedSegments.append(resegment(s))
        return returnedSegments
                    
    else:
        return segment

if __name__ == "__main__":
    if len(sys.argv)!=1:
        print ("xml src and/or image src not specified")
    else:
#        src_xml_file = sys.argv[1]
#        src_image = sys.argv[2]
#        detection(src_xml_file,src_image)
        src_xml_file = "../../processed_pdfs/alphabet1/alphabet1.xml"
        src_image = "../../processed_pdfs/alphabet1/alphabet1-1.png"
        detection(src_xml_file,src_image)