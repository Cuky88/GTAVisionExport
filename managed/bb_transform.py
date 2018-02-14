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

def ff(nmask, depth, seeds, lL, uL):
    h,w = depth.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    mask[1:h+1, 1:w+1] = nmask

    floodflags = 8
    #floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (128 << 8)
    #floodflags |= cv2.FLOODFILL_FIXED_RANGE

    rects = list()
    for seed in seeds:
        #num,im,mask,rect = cv2.floodFill(im, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)
        num,im,mask,rect = cv2.floodFill(depth, mask, seed, 255, lL, uL, floodflags)

        # draw a blue circle at the seed point
        cv2.circle(depth, seed, 1, 0, 2)
        cv2.circle(mask, seed, 1, 255, 2)

        # Check if all rects are correct, if not, delete them
        if rect[2] is not 0 and rect[3] is not 0:
            rects.append(rect)

    return im, mask, rects

def maskOff(id, im):
    # This function is needed to manipulate the mask, so that it can be used in the main flood fill function (ff)
    # Basically this creates the same mask as the input but only for a certain category id; The original mask is somehow not usable in cv2, so flood fill is used to convert the mask
    a = np.array(im)
    #print(np.unique(a)) # [0 1 2 3 4 7 8]

    # Create mask for object:
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
    # Notice the size needs to be 2 pixels larger than the image.
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
            if dist == 1.0:# or dist == 0.0:
                accepted.append(seed)
                break

    return accepted

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
    
if __name__ == '__main__':
    # python bb_transform.py --mask 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', default=None, type=int)
    parser.add_argument('--iou', default=0.5, type=float)
    parser.add_argument('--lL', default=0.005, type=float)
    parser.add_argument('--uL', default=0.05, type=float)
    parser.add_argument('--debug', default=0, type=int)
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
            # Read stencil image
            im_s = cv2.imread("Data\\" + img["Image"].split(".")[0] + stencil_s)
            # Mask array of cars set to 255, rest to 0
            mask = maskOff(args.mask, im_s)
            # Stencil is in 8-bit int
            #mask.astype(np.uint8)

            # Read depth image
            im_d = np.array(tifffile.imread("Data\\" + img["Image"].split(".")[0] + depth_s))
            # Convert float to int
            maxF = np.max(im_d)
            im_d = 1*im_d/maxF
            #im_d = im_d.astype(np.uint8)

            # Mask depth image based from stencil and remove non-car pixels
            im_d_masked = cv2.bitwise_and(im_d, im_d, mask=mask)

            if args.debug:
                print("Data\\" + img["Image"].split(".")[0] + depth_s)
                cut_d_masked = im_d_masked
                cut_mask = mask

                # Check if array is all zero, if so, skip this
                if not np.any(cut_d_masked):
                    continue

                # Calculate contour and center of stencil blobs and draw bounding box
                contours, centers, bb = findContours(cut_mask)
                # Create new mask with stencil blobs, since stencil image cannot be used due to data format (8-bit)
                nmask = np.zeros((cut_mask.shape[0], cut_mask.shape[1]), np.uint8)
                # Draw contours on new mask; those will be the boundaries for the flood fill
                cv2.drawContours(nmask, contours, -1, (255, 255, 255), 1)
                # Divide the bounding boxes of the stencil blobs and calculate for each new rectangle the centers, which will be added to the seed points later
                divider = 4
                bbcenters = divideBB(bb, divider)
                # Check if the new seed points are lying in the contours and add them to the list
                acceptSeeds = centerInPoly(bbcenters, contours)

                print("cut_d_masked")
                cv2.imshow("Test", cut_d_masked)
                key = cv2.waitKey(0)
                if key == 27: # escape
                    cv2.destroyAllWindows()
                    break

                allSeeds = list()
                # Add center of stencil blobs as seeds and also the centers of the divided bounding boxes
                allSeeds.extend(centers)
                allSeeds.extend(acceptSeeds)

                if not allSeeds:
                    continue

                # Visualize the seeds
                for b in acceptSeeds:
                    cv2.circle(nmask, b, 2, (255, 255, 255), -1)

                print("nmask")
                cv2.imshow("Test", nmask)
                key = cv2.waitKey(0)
                if key == 27: # escape
                    cv2.destroyAllWindows()
                    break

                # Do flood fill on the masked depth image with all the seeds
                ffim, ffmask, rects = ff(nmask, cut_d_masked, allSeeds, args.lL, args.uL)

                print("ffim")
                cv2.imshow("Test", ffim)
                key = cv2.waitKey(0)
                if key == 27: # escape
                    cv2.destroyAllWindows()
                    break
                    
                # Draw adjusted bounding boxes on flood filled mask
                ffmask = cv2.cvtColor(ffmask,cv2.COLOR_GRAY2RGB)

                for r in rects:
                    print("BBnew: (%d, %d, %d, %d)"%(r[0], r[1], r[0]+r[2], r[1]+r[3]))
                    ffmask = cv2.rectangle(ffmask, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 255, 0), 1)

                ffmask = cv2.applyColorMap(ffmask, cv2.COLORMAP_JET)
                print("ffmask")
                cv2.imshow("Test", ffmask)
                key = cv2.waitKey(0)

                if key == 27: # escape
                    cv2.destroyAllWindows()
                    break
                continue
                    
            for det in img["Detections"]:
                if det["Type"] == "car" and det["Visibility"] == True:
                    print("Data\\" + img["Image"].split(".")[0] + depth_s)

                    # Get old bounding box from json as (minx, miny, width, height)
                    minx, miny, w, h = det["BBmin"]["X"], det["BBmin"]["Y"], det["BBmax"]["X"] - det["BBmin"]["X"], det["BBmax"]["Y"] - det["BBmin"]["Y"]

                    # Cut out mask and depth images based on the old bounding box for this car
                    cut_d_masked = im_d_masked[miny:miny+h, minx:minx+w]
                    cut_mask = mask[miny:miny+h, minx:minx+w]

                    # Check if array is all zero, if so, skip this
                    if not np.any(cut_d_masked):
                        continue

                    # Calculate contour and center of stencil blobs and draw bounding box
                    contours, centers, bb = findContours(cut_mask)
                    # Create new mask with stencil blobs, since stencil image cannot be used due to data format (8-bit)
                    nmask = np.zeros((cut_mask.shape[0], cut_mask.shape[1]), np.uint8)
                    # Draw contours on new mask; those will be the boundaries for the flood fill
                    cv2.drawContours(nmask, contours, -1, (255, 255, 255), 1)
                    # Divide the bounding boxes of the stencil blobs and calculate for each new rectangle the centers, which will be added to the seed points later
                    if w >= 150 or h >= 150:
                        divider = 10
                    else:
                        divider = 4
                    bbcenters = divideBB(bb, divider)
                    # Check if the new seed points are lying in the contours and add them to the list
                    acceptSeeds = centerInPoly(bbcenters, contours)

                    print("cut_d_masked")
                    cv2.imshow("Test", cut_d_masked)
                    key = cv2.waitKey(0)
                    if key == 27: # escape
                        cv2.destroyAllWindows()
                        break

                    allSeeds = list()
                    # Add center of stencil blobs as seeds and also the centers of the divided bounding boxes
                    allSeeds.extend(centers)
                    allSeeds.extend(acceptSeeds)

                    if not allSeeds:
                        continue

                    # Visualize the seeds
                    for b in acceptSeeds:
                        cv2.circle(nmask, b, 2, (255, 255, 255), -1)

                    print("nmask")
                    cv2.imshow("Test", nmask)
                    key = cv2.waitKey(0)
                    if key == 27: # escape
                        cv2.destroyAllWindows()
                        break

                    # Do flood fill on the masked depth image with all the seeds
                    ffim, ffmask, rects = ff(nmask, cut_d_masked, allSeeds, args.lL, args.uL)

                    print("ffim")
                    cv2.imshow("Test", ffim)
                    key = cv2.waitKey(0)
                    if key == 27: # escape
                        cv2.destroyAllWindows()
                        break
                    
                    # Draw adjusted bounding boxes on flood filled mask
                    ffmask = cv2.cvtColor(ffmask,cv2.COLOR_GRAY2RGB)
                    print("BBold: (%d, %d, %d, %d)"%(minx, miny, w, h))

                    if len(rects) is 1:
                        r = rects[0]
                        print("BBnew: (%d, %d, %d, %d)"%(minx+r[0], miny+r[1], r[2], r[3]))
                        bb1 = {'x1':minx, 'x2':minx+w, 'y1':miny, 'y2':miny+h}
                        bb2 = {'x1':minx+r[0], 'x2':minx+r[0]+r[2], 'y1':miny+r[1], 'y2':miny+r[1]+r[3]}
                        ffmask = cv2.rectangle(ffmask, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 255, 0), 1)
                        print("IOU: %f"%get_iou(bb1, bb2))
                    else:
                        print("More then 1 Bounding Box found")


                    ffmask = cv2.applyColorMap(ffmask, cv2.COLORMAP_JET)
                    print("ffmask")
                    cv2.imshow("Test", ffmask)
                    key = cv2.waitKey(0)

                    if key == 27: # escape
                        cv2.destroyAllWindows()
                        break
    else:
        print("[ERROR] No mask id given!")

    cv2.destroyAllWindows()
