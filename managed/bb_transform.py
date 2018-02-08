import os
import sys
import numpy as np
from PIL import Image, ImageFile
from skimage import io
from matplotlib import cm
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tifffile
from os import listdir, makedirs
from os.path import isfile, join
import json
import cv2
import sys
import argparse
import heapq
import imutils

# Test with image
# gtav_cid0_c258_1.tiff
# gtav_cid0_c258_1-depth.tiff
# gtav_cid0_c258_1-stencil.tiff

def ff(nmask, depth, seeds):
    h,w = depth.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    mask[1:h+1, 1:w+1] = nmask

    floodflags = 4
    #floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (128 << 8)
    uL = 0.009
    lL = 0.00099

    rects = list()
    for seed in seeds:
        #num,im,mask,rect = cv2.floodFill(im, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)
        num,im,mask,rect = cv2.floodFill(depth, mask, seed, 255, lL, uL, floodflags)

        # draw a blue circle at the seed point
        cv2.circle(depth, seed, 1, 0, 2)
        cv2.circle(mask, seed, 1, 255, 2)
        rects.append(rect)

    return im, mask, rects

def maskOff(id, im):
    a = np.array(im)
    #print(np.unique(a)) # [0 1 2 3 4 7 8]

    # Create mask
    # a[np.logical_or(a>8, a<8)] = 0 # For Plants?
    # a[np.logical_or(a>7, a<7)] = 0 # For Buildings and Objects
    # a[np.logical_or(a>6, a<6)] = 0  # Empty!
    # a[np.logical_or(a>5, a<5)] = 0 # Empty!
    # a[np.logical_or(a>4, a<4)] = 0 # Undefined!
    # a[np.logical_or(a>3, a<3)] = 0 # Plants
    # a[np.logical_or(a>2, a<2)] = 0 # Cars
    # a[np.logical_or(a>1, a<1)] = 0 # People

    a[np.logical_or(a>id, a<id)] = 0

    # Copy the thresholded image.
    im_floodfill = a.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = a.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = a | im_floodfill_inv

    return im_out[:,:,0]

def findContours(mask, im=None):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    centers = list()
    bb = list()
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0.0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    
        # draw the contour and center of the shape on the image
        #cv2.drawContours(im, [c], -1, (0, 255, 0), 2)
        #cv2.circle(im, (cX, cY), 2, (0, 0, 255), -1)

        x,y,w,h = cv2.boundingRect(c)
        bb.append((x,y,w,h))
        centers.append((cX, cY))
        
    # Show keypoints
    # cv2.imshow("contours", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return cnts, centers, bb

def divideBB(bb, div):
    '''
    If div == 2:
    +-------------+-------------+
    |             |             |
    |             |             |
    |-------------+-------------+
    |             |             |
    |             |             |
    +-------------+-------------+  

    Get centers of the boxes and add them to seed points later!          
    '''
    dividedCenters = list()
    for box in bb:
        minx = box[0]
        miny = box[1]
        maxx = box[0] + box[2]
        maxy = box[1] + box[3]

        l = box[2]/div
        h = box[3]/div

        for i in range(div):
            for j in range(div):
                cX = minx + (l * i) + l/2
                cY = miny + (h * j) + h/2
                dividedCenters.append((int(cX), int(cY)))

    return dividedCenters

def centerInPoly(bbcenters, contours):
    accepted = list()

    for seed in bbcenters:
        for cnt in contours:
            dist = cv2.pointPolygonTest(cnt,seed,False)
            if dist == 1.0 or dist == 0.0:
                accepted.append(seed)
                break

    return accepted
    
if __name__ == '__main__':
    # python bb_transform.py --mask 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', default=None, type=int)
    args = parser.parse_args()
    stencil_s = "-stencil.tiff"
    depth_s = "-depth.tiff"

    cv2.namedWindow("Test")        # Create a named window
    cv2.moveWindow("Test", 40,30)  # Move it to (40,30)

    if  args.mask is not None:
        data = []
        with open("data_boxes.json") as reader:
            read = json.load(reader)
            for line in read:
                data.append(line)

        for img in data:
            #im = cv2.imread("Data\\" + img["Image"])
            im_s = cv2.imread("Data\\" + img["Image"].split(".")[0] + stencil_s)
            # Mask array of cars set to 255, rest to 0
            m = maskOff(args.mask, im_s)
            m.astype(np.uint8)

            #imcol = cv2.applyColorMap(m, cv2.COLORMAP_JET)
            # cv2.imshow("Mask", m)
            # key = cv2.waitKey(0)
            # cv2.destroyAllWindows()

            contours, centers, bb = findContours(m)
            nmask = np.zeros((m.shape[0], m.shape[1]), np.uint8)
            cv2.drawContours(nmask, contours, -1, (255, 255, 255), 1)
            bbcenters = divideBB(bb, 4)
            acceptSeeds = centerInPoly(bbcenters, contours)

            #im_d = cv2.imread("Data\gtav_cid0_c258_1-depth.tiff")
            im_d = np.array(tifffile.imread("Data\\" + img["Image"].split(".")[0] + depth_s))
            maxF = np.max(im_d)
            im_d = 1*im_d/maxF
            #im_d = im_d.astype(np.uint8)

            # cv2.imshow("Depth", im_d)
            # key = cv2.waitKey(0)
            # cv2.destroyAllWindows()

            allSeeds = list()
            # Get Position of detections also
            # for det in img["Detections"]:
            #     if det["Type"] == "car" and det["Visibility"] == True:
            #         if 0 <= det["Pos2D"]["X"] <= m.shape[0] and 0 <= det["Pos2D"]["Y"] <= m.shape[1]:
            #             allSeeds.append((det["Pos2D"]["X"], det["Pos2D"]["Y"]))

            allSeeds.extend(centers)
            allSeeds.extend(acceptSeeds)

            ffim, ffmask, rects = ff(nmask, im_d, allSeeds)

            for b in acceptSeeds:
                cv2.circle(nmask, b, 2, (255, 255, 255), -1)

            cv2.imshow("Test", nmask)
            key = cv2.waitKey(0)
            if key == 27: # escape
                cv2.destroyAllWindows()
                break

            cv2.putText(ffim,"Data\\" + img["Image"].split(".")[0] + depth_s, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow("Test", ffim)
            key = cv2.waitKey(0)
            if key == 27: # escape
                cv2.destroyAllWindows()
                break
            
            ffmask = cv2.cvtColor(ffmask,cv2.COLOR_GRAY2RGB)
            for r in rects:
                ffmask = cv2.rectangle(ffmask, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 255, 0), 1)

            ffmask = cv2.applyColorMap(ffmask, cv2.COLORMAP_JET)
            cv2.imshow("Test", ffmask)
            key = cv2.waitKey(0)

            if key == 27: # escape
                cv2.destroyAllWindows()
                break
    else:
        print("[ERROR] No mask id given!")

    cv2.destroyAllWindows()
