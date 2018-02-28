import os
import sys
import numpy as np
from PIL import Image, ImageFile, ImageDraw, ImageFont
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
import shutil
import glob
import re
from subprocess import call
import math
import bb_transform as bbt
import itertools
from pathlib import Path
from tqdm import tqdm

# Python program to implement Cohen Sutherland algorithm
# for line clipping.
 
# Defining region codes
INSIDE = 0  #0000
LEFT = 1    #0001
RIGHT = 2   #0010
BOTTOM = 4  #0100
TOP = 8     #1000

depths = {}
stencils = {}
in_directory = 'Data\\'
out_directory = 'Data\\img2\\'
DEBUG_TRANS = False


# Class for creating rectangle objects; is needed to check if one rectangle is in another rectangle
class Rectangle:
    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1<x2 and y1<y2:
            return type(self)(x1, y1, x2, y2)
    __and__ = intersection

    def difference(self, other):
        inter = self&other
        if not inter:
            yield self
            return
        xs = {self.x1, self.x2}
        ys = {self.y1, self.y2}
        if self.x1<other.x1<self.x2: xs.add(other.x1)
        if self.x1<other.x2<self.x2: xs.add(other.x2)
        if self.y1<other.y1<self.y2: ys.add(other.y1)
        if self.y1<other.y2<self.y2: ys.add(other.y2)
        for (x1, x2), (y1, y2) in itertools.product(
            pairwise(sorted(xs)), pairwise(sorted(ys))
        ):
            rect = type(self)(x1, y1, x2, y2)
            if rect!=inter:
                yield rect
    __sub__ = difference

    def __init__(self, x1, y1, x2, y2):
        if x1>x2 or y1>y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def __eq__(self, other):
        return isinstance(other, Rectangle) and tuple(self)==tuple(other)
    def __ne__(self, other):
        return not (self==other)

    def __repr__(self):
        return type(self).__name__+repr(tuple(self))


def pairwise(iterable):
    # https://docs.python.org/dev/library/itertools.html#recipes
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

# Function to compute region code for a point(x,y)
def computeCode(x, y, x_max, y_max, x_min, y_min):
    code = INSIDE
    if x < x_min:      # to the left of rectangle
        code |= LEFT
    elif x > x_max:    # to the right of rectangle
        code |= RIGHT
    if y < y_min:      # below the rectangle
        code |= BOTTOM
    elif y > y_max:    # above the rectangle
        code |= TOP
 
    return code
 
# Implementing Cohen-Sutherland algorithm
# Clipping a line from P1 = (x1, y1) to P2 = (x2, y2)
def cohenSutherlandClip(p1, p2, imw, imh):
    # Defining x_max,y_max and x_min,y_min for rectangle
    # Since diagonal points are enough to define a rectangle
    x_max = imw
    y_max = imh
    x_min = 0.0
    y_min = 0.0

    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    # Compute region codes for P1, P2
    code1 = computeCode(x1, y1, x_max, y_max, x_min, y_min)
    code2 = computeCode(x2, y2, x_max, y_max, x_min, y_min)
    accept = False
    bothOut = False
    clipped = False

    while True:
        # If both endpoints lie within rectangle
        if code1 == 0 and code2 == 0:
            accept = True
            break
 
        # If both endpoints are outside rectangle
        elif (code1 & code2) != 0:
            bothOut = True
            break
 
        # Some segment lies within the rectangle
        else:
            clipped = True
            # Line Needs clipping
            # At least one of the points is outside, 
            # select it
            x = 1.0
            y = 1.0
            if code1 != 0:
                code_out = code1
            else:
                code_out = code2
 
            # Find intersection point
            # using formulas y = y1 + slope * (x - x1), 
            # x = x1 + (1 / slope) * (y - y1)
            if code_out & TOP:
               
                # point is above the clip rectangle
                x = x1 + (x2 - x1) * \
                                (y_max - y1) / (y2 - y1)
                y = y_max
 
            elif code_out & BOTTOM:
                 
                # point is below the clip rectangle
                x = x1 + (x2 - x1) * \
                                (y_min - y1) / (y2 - y1)
                y = y_min
 
            elif code_out & RIGHT:
                 
                # point is to the right of the clip rectangle
                y = y1 + (y2 - y1) * \
                                (x_max - x1) / (x2 - x1)
                x = x_max
 
            elif code_out & LEFT:
                 
                # point is to the left of the clip rectangle
                y = y1 + (y2 - y1) * \
                                (x_min - x1) / (x2 - x1)
                x = x_min
 
            # Now intersection point x,y is found
            # We replace point outside clipping rectangle
            # by intersection point
            if code_out == code1:
                x1 = x
                y1 = y
                code1 = computeCode(x1,y1, x_max, y_max, x_min, y_min)
 
            else:
                x2 = x
                y2 = y
                code2 = computeCode(x2, y2, x_max, y_max, x_min, y_min)
 
    if accept or clipped:
        if DEBUG_TRANS:
            print ("Line accepted from %.2f,%.2f to %.2f,%.2f" % (x1,y1,x2,y2))
        return((x1,y1), (x2,y2)) 
    else:
        if DEBUG_TRANS:
            print("Line rejected")
        return None

def rotate(p, theta):
    # Rotation order: Z Y X
    X = np.cos(theta[2]) * (np.cos(theta[1]) * p[0] + np.sin(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) -  np.sin(theta[2]) * (np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    #X = np.cos(theta[2]) * ((np.cos(theta[1]) * p[0] + np.sin(theta[1]) * ((np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) - np.sin(theta[2]) * ((np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    Y = np.sin(theta[2]) * (np.cos(theta[1]) * p[0] + np.sin(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) + np.cos(theta[2]) * (np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    #Y = np.sin(theta[2]) * (np.cos(theta[1]) * p[0] + np.sin(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) + np.cos(theta[2]) * (np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    Z = -np.sin(theta[1]) * p[0] + np.cos(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])
    #Z = -np.sin(theta[1]) * p[0] + np.cos(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])
    
    if DEBUG_TRANS:
        print(X, Y, Z)
    return np.array([X,Y,Z])

def getDotVectorResult(camPos, corner, camForward):
    dir = corner - camPos
    dirNorm = dir/ np.linalg.norm(dir)
    pos =  np.dot(dirNorm, camForward)

    if DEBUG_TRANS:
        print("##################################################")
        print(corner, camPos)
        print("dirNorm: %s pos: %s"%(str(dirNorm), str(pos)))

    if (pos >= 0):
        # If vertices is in front of cam, the normalized distance will be returned
        return pos
    else:
        # If vertices is behind of cam, -1 will be returned
        return -1.0

def convertToCamCord(pList, Camrot, Campos, camNearClip, camFOV, imgw, imgh, uiw, uih):
    pListProj = list()
    for p in pList:
        # camera rotation 
        theta = (np.pi / 180) * np.array([Camrot["X"],Camrot["Y"], Camrot["Z"]], dtype=np.float64)

        # camera direction, at 0 rotation the camera looks down the postive Y axis --> WorldNorth schaut somit immer in Cam-Richtung
        camDir = rotate(np.array([0,1,0]), theta)

        # Check if point is behind the camera
        sign = 1.0
        scale = 1.0
        if (getDotVectorResult(np.array([Campos["X"],Campos["Y"], Campos["Z"]], dtype=np.float64), p, camDir) == -1):
            sign = -1.0
            scale = 15
            # If point is behind the cam, the camera direction will be inverted
            camDir = sign * rotate(np.array([0,1,0]), theta)

        # camera position  == eigentlich Bildebene rotiert in Blickrichtung, minimal vor der Cam zu sehen
        c = np.array([Campos["X"],Campos["Y"], Campos["Z"]], dtype=np.float64) + camNearClip * camDir

        # viewer position == Cam-Pos rotiert in Blickrichtung; minimal hinter c!
        e = -camNearClip * camDir

        viewWindowHeight = 2 * camNearClip * np.tan((camFOV / 2) * (np.pi / 180))
        viewWindowWidth = (imgw*1. / imgh*1.) * viewWindowHeight
        
        camUp = rotate(np.array([0,0,1]), theta)
        
        camEast = rotate(np.array([1,0,0]), theta)

        # Distanz zwischen Punkt und Bildebene
        delete = p - c
        
        viewerDist = delete - e
        
        # Vector3 viewerDistNorm = viewerDist * (1 / viewerDist.Length());
        viewerDistNorm = viewerDist/np.linalg.norm(viewerDist)
        
        dot = np.dot(camDir, viewerDistNorm)
        ang = np.arccos(dot)
        
        # Senkrechter Abstand zur Bildebene
        viewPlaneDist = camNearClip / np.cos(ang)

        # Falls Punkt hinter der Kamera liegt, wird der Abstand zur Bildebene vergrößert; sonst erscheint der Punkt auf dem Bild, wenn es lange Fahrzeuge sind
        viewPlaneDist = viewPlaneDist * scale
        
        # Punkt auf der Bildebene
        viewPlanePointTMP = viewPlaneDist * viewerDistNorm + e

        # move origin to upper left 
        newOrigin = c + (viewWindowHeight / 2) * camUp - (viewWindowWidth / 2) * camEast
        viewPlanePoint = (viewPlanePointTMP + c) - newOrigin

        viewPlaneX = np.dot(viewPlanePoint, camEast) / np.dot(camEast, camEast)
        viewPlaneZ = np.dot(viewPlanePoint, camUp) / np.dot(camUp, camUp)

        screenX = viewPlaneX / viewWindowWidth * uiw
        screenY = -viewPlaneZ / viewWindowHeight * uih
        
        Xscale = float(imgw) / (1.0 * uiw) * screenX
        Yscale = float(imgh) / (1.0 * uih) *screenY

        if DEBUG_TRANS:
            print("Theta: %s"%str(theta))
            print("camDir: %s"%str(camDir))
            print("c: %s"%str(c))
            print("e: %s"%str(e))
            print("viewWindowHeight: %s"%str(viewWindowHeight))
            print("viewWindowWidth: %s"%str(viewWindowWidth))
            print("camUp: %s"%str(camUp))
            print("camEast: %s"%str(camEast))
            print("delete: %s"%str(delete))
            print("viewerDist: %s"%str(viewerDist))
            print("viewerDistNorm: %s"%str(viewerDistNorm))
            print("dot: %s"%str(dot))
            print("ang: %s"%str(ang))
            print("viewPlaneDist: %s"%str(viewPlaneDist))
            print("viewPlanePoint: %s"%str(viewPlanePointTMP))
            print("newOrigin: %s"%str(newOrigin))
            print("viewPlanePoint: %s"%str(viewPlanePoint))
            print("viewPlaneX: %s"%str(viewPlaneX))
            print("viewPlaneZ: %s"%str(viewPlaneZ))
            print("screenX: %s"%str(screenX))
            print("screenY: %s"%str(screenY))
            print("Xscale: %s"%str(Xscale))
            print("Yscale: %s"%str(Yscale))

        #pListProj.append((int(Xscale), int(Yscale)))
        pListProj.append((np.asscalar(np.array([Xscale], dtype=np.int64)), np.asscalar(np.array([Yscale], dtype=np.int64))))
        
    return pListProj

def get2DBB(pListProj, imgw, imgh, uiw, uih):
    # Order of 3d bb points
    #FURGame - 0
    #FULGame - 1
    #BULGame - 2
    #BURGame - 3
    #FLLGame - 4
    #BLLGame - 5
    #BLRGame - 6
    #FLRGame - 7

    pListBB = []
    clipped = []
    clipp = cohenSutherlandClip(pListProj[0], pListProj[1], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])
    clipp = cohenSutherlandClip(pListProj[1], pListProj[2], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])
    clipp = cohenSutherlandClip(pListProj[2], pListProj[3], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])
    clipp = cohenSutherlandClip(pListProj[3], pListProj[0], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])

    clipp = cohenSutherlandClip(pListProj[4], pListProj[5], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])
    clipp = cohenSutherlandClip(pListProj[5], pListProj[6], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])
    clipp = cohenSutherlandClip(pListProj[6], pListProj[7], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])
    clipp = cohenSutherlandClip(pListProj[7], pListProj[4], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])

    clipp = cohenSutherlandClip(pListProj[0], pListProj[7], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])
    clipp = cohenSutherlandClip(pListProj[1], pListProj[4], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])
    clipp = cohenSutherlandClip(pListProj[2], pListProj[5], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])
    clipp = cohenSutherlandClip(pListProj[3], pListProj[6], imgw, imgh)
    if clipp is not None:
        clipped.append(clipp[0])
        clipped.append(clipp[1])

    # Get the min and max values of the 2D BB based on the clipped corners of the 3D BB
    minX = 0
    minY = 0
    maxX = 0
    maxY = 0

    unzipped = list(zip(*clipped))
    if unzipped:
        minX = np.min(unzipped[0])
        maxX = np.max(unzipped[0])
        minY = np.min(unzipped[1])
        maxY = np.max(unzipped[1])

    pListBB.append([(np.asscalar(np.array([minX], dtype=np.int64)), np.asscalar(np.array([minY], dtype=np.int64))), 
        (np.asscalar(np.array([maxX], dtype=np.int64)), np.asscalar(np.array([maxY], dtype=np.int64)))])
    return pListBB

def bbox_from_string(name):
    bbox2 = []
    bbox3 = []
    for d in data:
        if d['Image'] == name:
            print( d['Campos'])
            for i,p in enumerate(d['Detections']):
                if p["Type"] == "car" and p["Visibility"] == True:
                    pList = list()
                    pList.append(np.array([p["FURGame"]["X"],p["FURGame"]["Y"],p["FURGame"]["Z"]]))
                    pList.append(np.array([p["FULGame"]["X"],p["FULGame"]["Y"],p["FULGame"]["Z"]]))
                    pList.append(np.array([p["BULGame"]["X"],p["BULGame"]["Y"],p["BULGame"]["Z"]]))
                    pList.append(np.array([p["BURGame"]["X"],p["BURGame"]["Y"],p["BURGame"]["Z"]]))
                    pList.append(np.array([p["FLRGame"]["X"],p["FLRGame"]["Y"],p["FLRGame"]["Z"]]))
                    pList.append(np.array([p["FLLGame"]["X"],p["FLLGame"]["Y"],p["FLLGame"]["Z"]]))
                    pList.append(np.array([p["BLLGame"]["X"],p["BLLGame"]["Y"],p["BLLGame"]["Z"]]))
                    pList.append(np.array([p["BLRGame"]["X"],p["BLRGame"]["Y"],p["BLRGame"]["Z"]]))
                    BB3D, BB2D = convertToCamCord(pList, d['Camrot'], d['Campos'], d['CamNearClip'], d['CamFOV'], d['ImageWidth'], d['ImageHeight'], d['UIwidth'], d['UIheight'])
                    bbox3.append(BB3D)        
                    bbox2.append(BB2D) 
    return bbox3, bbox2

def show_bounding_boxes(name, size, ax, data):
    #FURGame - 0
    #FULGame - 1
    #BULGame - 2
    #BURGame - 3
    #FLRGame - 4
    #FLLGame - 5
    #BLLGame - 6
    #BLRGame - 7
    
    bbox2 = []
    for d in data:
        if d['Image'] == name:
            for i,p in enumerate(d['Detections']):
                print(p["Visibility"])
                if p["Type"] == "car" and p["Visibility"]:
                    pList = list()
                    pList.append(np.array([p["FUR"]["X"],p["FUR"]["Y"]]))
                    pList.append(np.array([p["FUL"]["X"],p["FUL"]["Y"]]))
                    pList.append(np.array([p["BUL"]["X"],p["BUL"]["Y"]]))
                    pList.append(np.array([p["BUR"]["X"],p["BUR"]["Y"]]))
                    pList.append(np.array([p["FLR"]["X"],p["FLR"]["Y"]]))
                    pList.append(np.array([p["FLL"]["X"],p["FLL"]["Y"]]))
                    pList.append(np.array([p["BLL"]["X"],p["BLL"]["Y"]]))
                    pList.append(np.array([p["BLR"]["X"],p["BLR"]["Y"]]))

                    bbox2.append(pList) 

    codes1 = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
        ]
        
    codes2 = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
        ]
         
    codes3 = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
        ]
        
    codes4 = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
        ]
        
    for c in bbox2:
        x, y = zip(*c)
        ax.plot(x,y, 'yo')
        
        path1 = Path([c[0], c[1], c[2], c[3], c[0]], codes1)
        path2 = Path([c[4], c[5], c[6], c[7], c[4]], codes2)
        path3 = Path([c[0], c[4], c[7], c[3], c[0]], codes3)
        path4 = Path([c[1], c[5], c[6], c[2], c[1]], codes4)
        patch1 = patches.PathPatch(path1, facecolor="none", lw=1, edgecolor='blue')
        patch2 = patches.PathPatch(path2, facecolor="none", lw=1, edgecolor='blue')
        patch3 = patches.PathPatch(path3, facecolor="none", lw=1, edgecolor='blue')
        patch4 = patches.PathPatch(path4, facecolor="none", lw=1, edgecolor='blue')
        ax.add_patch(patch1)
        ax.add_patch(patch2)
        ax.add_patch(patch3)
        ax.add_patch(patch4)
    return

def load_depth(name):
    if name not in depths:
        tiff_depth = tifffile.imread(os.path.join(in_directory, name.split(".")[0] + '-depth.tiff'))
        depths[name.split(".")[0] + '-depth.tiff'] = tiff_depth
    return depths[name.split(".")[0] + '-depth.tiff']


def load_stencil(name):
    if name not in stencils:
        tiff_stencil = tifffile.imread(os.path.join(in_directory, name.split(".")[0] + '-stencil.tiff'))
        stencils[name.split(".")[0] + '-stencil.tiff'] = tiff_stencil
    return stencils[name.split(".")[0] + '-stencil.tiff']


def load_stencil_ids(name):
    stencil = load_stencil(name)
    return stencil % 16  # only last 4 bits are object ids


def load_stencil_flags(name):
    stencil = load_stencil(name)
    return stencil - (stencil % 16)  # only first 4 bits are flags


def ids_to_greyscale(arr):
    # there are 4 bits -> 16 values for arrays, transfer from range [0-15] to range [0-255]
    return arr * 100


def main():
    files = [f for f in listdir(in_directory) if isfile(join(in_directory, f)) and f.split(".")[-1]=="tiff" and len(f.split("-"))==1]
    
    data = []
    with open("data_boxes.json") as reader:
        read = json.load(reader)
        for line in read:
            data.append(line)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for name in files:
        print(name)

        #img = cv2.imread(os.path.join(in_directory, name.split(".")[0] + "-depth.tiff"),  cv2.IMREAD_UNCHANGED)

        #plt.imshow(img)
        #plt.show()

        im = Image.open(os.path.join(in_directory, name))
        size = (im.size[1], im.size[0])

        #fig = plt.figure()
        #show_3dbounding_boxes(im, name, size, ax1)
        #plt.imshow(im2)
        # show_bounding_boxes(name, size, plt.gca())
        # plt.savefig(os.path.join(out_directory, 'bboxes-' + name + '.jpg'))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        plt.tight_layout()

        plt.axis('off')
        patch = patches.Rectangle((0,0), 1280, 720, fill=False)
        ax1.set_clip_path(patch)
        colim = ax1.imshow(im, clip_on=True)
        # ax2.set_title('f')
        ax2.imshow(ids_to_greyscale(load_stencil_ids(name)), cmap='gray', clip_on=True)
        ax3.set_title('ids')
        # ax3.imshow(load_stencil_ids(name), cmap='gray')
        ax3.imshow(ids_to_greyscale(load_stencil_ids(name)), cmap='plasma', clip_on=True)
        ax4.set_title('depth')
        ax4.imshow(load_depth(name), cmap='gray', clip_on=True)

        # Der funzt einigermassen gut; Allg. Punktezeichnen bei Matplotlib ist nicht sonderlich gut
        # show_bounding_boxes(name, size, ax1, data)

        plt.axis('off')
        plt.draw()
        plt.show()

    plt.show()

def plotCV(mode, iou, minw, minh):
    files = [f for f in listdir(in_directory) if isfile(join(in_directory, f)) and f.split(".")[-1]=="tiff" and len(f.split("-"))==1]

    data = []
    with open("data_boxes.json") as reader:
        read = json.load(reader)
        for line in read:
            data.append(line)
    
    cv2.namedWindow("Test")        # Create a named window
    cv2.moveWindow("Test", 40,30)  # Move it to (40,30)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for name in files:       
        for d in data:
            # gtav_cid1_c-1_38.tiff
            if d['Image'] == name:
                imname = 'D:\\Devel\\GTAVisionExport\\managed\\Data\\' + name
                print(imname)
                imgPil = Image.open(imname)

                # cv2.cvtColor macht am FZI Rechner Probleme!
                im = cv2.cvtColor(np.array(imgPil), cv2.COLOR_BGR2RGB)
                img = np.array(im)

                #if img == None: print("Image %s not found!"%('D:\\Devel\\GTAVisionExport\\managed\\Data\\' + name)); break

                for p in d['Detections']:
                    if p["Type"] == "car" and p["Visibility"] == True:
                        
                        # If old bounding box is too small, don't change it
                        #if w < minw or h < minh:
                        #   img["Detections"][i] = det
                        #    continue

                        print("\n2D-Center: ")
                        print((p["Pos2D"]["X"], p["Pos2D"]["Y"]))
                        if 'BBminNew' in p and p["IOU"] > iou:
                            print("2D-BB NEW: ")
                            print((p["BBminNew"]["X"], p["BBminNew"]["Y"]), (p["BBmaxNew"]["X"], p["BBmaxNew"]["Y"]))
                        else:
                            print("2D-BB OLD: ")
                            print((p["BBmin"]["X"], p["BBmin"]["Y"]), (p["BBmax"]["X"], p["BBmax"]["Y"]))
                        print("3D-BB: ")
                        print((p["FUR"]["X"],p["FUR"]["Y"]), (p["FUL"]["X"],p["FUL"]["Y"]), (p["BUL"]["X"],p["BUL"]["Y"]), (p["BUR"]["X"],p["BUR"]["Y"]), 
                        (p["FLL"]["X"],p["FLL"]["Y"]), (p["BLL"]["X"],p["BLL"]["Y"]), (p["BLR"]["X"],p["BLR"]["Y"]), (p["FLR"]["X"],p["FLR"]["Y"]))
                        print("\n---------------------------------------------------------------------")
                        
                        if mode == 2:
                            if 'BBminNew' in p and p["IOU"] > iou:
                                img = cv2.rectangle(img, (p["BBminNew"]["X"], p["BBminNew"]["Y"]), (p["BBmaxNew"]["X"], p["BBmaxNew"]["Y"]), (255, 255, 0), 1)
                                img = cv2.circle(im, (p["Pos2DNew"]["X"], p["Pos2DNew"]["Y"]), 1, (0, 0, 255), 2)
                            else:
                                img = cv2.rectangle(img, (p["BBmin"]["X"], p["BBmin"]["Y"]), (p["BBmax"]["X"], p["BBmax"]["Y"]), (255, 255, 0), 1)
                                img = cv2.circle(im, (p["Pos2D"]["X"], p["Pos2D"]["Y"]), 1, (0, 0, 255), 2)

                        elif mode == 3:
                            img = cv2.line(img, (p["FUR"]["X"], p["FUR"]["Y"]), (p["FUL"]["X"], p["FUL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["FUL"]["X"], p["FUL"]["Y"]), (p["BUL"]["X"], p["BUL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BUL"]["X"], p["BUL"]["Y"]), (p["BUR"]["X"], p["BUR"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BUR"]["X"], p["BUR"]["Y"]), (p["FUR"]["X"], p["FUR"]["Y"]), (255, 0, 255), 1)

                            img = cv2.line(img, (p["FLR"]["X"], p["FLR"]["Y"]), (p["FLL"]["X"], p["FLL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["FLL"]["X"], p["FLL"]["Y"]), (p["BLL"]["X"], p["BLL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BLL"]["X"], p["BLL"]["Y"]), (p["BLR"]["X"], p["BLR"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BLR"]["X"], p["BLR"]["Y"]), (p["FLR"]["X"], p["FLR"]["Y"]), (255, 0, 255), 1)

                            img = cv2.line(img, (p["FUR"]["X"], p["FUR"]["Y"]), (p["FLR"]["X"], p["FLR"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["FUL"]["X"], p["FUL"]["Y"]), (p["FLL"]["X"], p["FLL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BUL"]["X"], p["BUL"]["Y"]), (p["BLL"]["X"], p["BLL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BUR"]["X"], p["BUR"]["Y"]), (p["BLR"]["X"], p["BLR"]["Y"]), (255, 0, 255), 1)

                            img = cv2.circle(im, (p["Pos2D"]["X"], p["Pos2D"]["Y"]), 1, (0, 0, 255), 2)

                        elif mode == 1:
                            if 'BBminNew' in p and p["IOU"] > iou:
                                img = cv2.rectangle(img, (p["BBminNew"]["X"], p["BBminNew"]["Y"]), (p["BBmaxNew"]["X"], p["BBmaxNew"]["Y"]), (0, 0, 255), 1)
                                img = cv2.circle(im, (p["Pos2DNew"]["X"], p["Pos2DNew"]["Y"]), 1, (0, 0, 255), 2)

                            img = cv2.rectangle(img, (p["BBmin"]["X"], p["BBmin"]["Y"]), (p["BBmax"]["X"], p["BBmax"]["Y"]), (0, 255, 0), 1)
                            img = cv2.circle(im, (p["Pos2D"]["X"], p["Pos2D"]["Y"]), 1, (0, 255, 0), 2)
                        else:
                            if 'BBminNew' in p and p["IOU"] > iou:
                                img = cv2.rectangle(img, (p["BBminNew"]["X"], p["BBminNew"]["Y"]), (p["BBmaxNew"]["X"], p["BBmaxNew"]["Y"]), (255, 255, 0), 1)
                                img = cv2.circle(im, (p["Pos2DNew"]["X"], p["Pos2DNew"]["Y"]), 1, (0, 0, 255), 2)
                            else:
                                img = cv2.rectangle(img, (p["BBmin"]["X"], p["BBmin"]["Y"]), (p["BBmax"]["X"], p["BBmax"]["Y"]), (255, 255, 0), 1)
                                img = cv2.circle(im, (p["Pos2D"]["X"], p["Pos2D"]["Y"]), 1, (0, 0, 255), 2)

                            img = cv2.line(img, (p["FUR"]["X"], p["FUR"]["Y"]), (p["FUL"]["X"], p["FUL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["FUL"]["X"], p["FUL"]["Y"]), (p["BUL"]["X"], p["BUL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BUL"]["X"], p["BUL"]["Y"]), (p["BUR"]["X"], p["BUR"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BUR"]["X"], p["BUR"]["Y"]), (p["FUR"]["X"], p["FUR"]["Y"]), (255, 0, 255), 1)

                            img = cv2.line(img, (p["FLR"]["X"], p["FLR"]["Y"]), (p["FLL"]["X"], p["FLL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["FLL"]["X"], p["FLL"]["Y"]), (p["BLL"]["X"], p["BLL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BLL"]["X"], p["BLL"]["Y"]), (p["BLR"]["X"], p["BLR"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BLR"]["X"], p["BLR"]["Y"]), (p["FLR"]["X"], p["FLR"]["Y"]), (255, 0, 255), 1)

                            img = cv2.line(img, (p["FUR"]["X"], p["FUR"]["Y"]), (p["FLR"]["X"], p["FLR"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["FUL"]["X"], p["FUL"]["Y"]), (p["FLL"]["X"], p["FLL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BUL"]["X"], p["BUL"]["Y"]), (p["BLL"]["X"], p["BLL"]["Y"]), (255, 0, 255), 1)
                            img = cv2.line(img, (p["BUR"]["X"], p["BUR"]["Y"]), (p["BLR"]["X"], p["BLR"]["Y"]), (255, 0, 255), 1)

                cv2.putText(img, name, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))        
                cv2.imshow("Test",img)

                key = cv2.waitKey(0)
                if key == 27: # escape
                    cv2.destroyAllWindows()
                    break

def createJson(off, maskID, lL, uL):
    stencil_s = "-stencil.tiff"
    depth_s = "-depth.tiff"
    
    border_msg("Reading object lists from %s"%in_directory)

    objlist = [f for f in listdir(in_directory) if isfile(join(in_directory, f)) and f.split(".")[-1]=="json"]
    data = []

    for obj in tqdm(objlist):
        print(join(in_directory, obj))
        with open(join(in_directory, obj)) as file:
            for line in file:
                data.append(json.loads(line))

    total = list()
    keyList = list()
    dic = {}
    missingFiles = 0
    optimizedBB = 0
    allBB = 0
    errorImg = 0

    border_msg("Starting processing of data...")
    print("Grouping in camera id and crossing id")
    for d in tqdm(data):
        # Check if image exists, if not, don't process the data;
        if not Path(join(in_directory, d['Image'])).is_file():
            missingFiles += 1
            continue

        # Filename: gtav_cid0_c2818_6.tiff
        camID = int(re.sub('[c]', '', d['Image'].split('_')[2])) # = 2818
        crossID = int(re.sub('[cid]', '', d['Image'].split('_')[1])) # 0
        uK = str(crossID) + "+" + str(camID)
        # Make unique key out of crossID and camID

        if uK not in keyList:
            keyList.append(uK)
            dic[uK] = [d]
        else:
            dic[uK].append(d)
    
    print("Starting main processing pipeline")
    for key, value in tqdm(dic.items()):
        for img in value:
            bbList = list()
            #print("%d < %d"%(value.index(img), len(value)-1))
            if value.index(img) < len(value)-1:
                # Filename: gtav_cid0_c2818_6.tiff
                crossID = int(re.sub('[cid]', '', img['Image'].split('_')[1])) # 0
                # key is camID
                # crossID = cid
                if (crossID is not 0 and int(key.split("+")[-1]) is -1):
                    errorImg += 1
                    continue
                
                num_frame = int(img['Image'].split('_')[3][:-5]) # = 6

                # 1. Create dic with key for every cam
                # 2. Sort files into the dict according to cam id
                # 3. Do transformation for every cam id seperately
                # 4. Discard last annotation for every cam, since no image is given

                if img['Image'] in []:
                    addNum = 2
                else:
                    addNum = 1

                if not off:
                    img['Image'] = "gtav_cid" + str(int(re.sub('[cid]', '', img['Image'].split('_')[1]))) + "_c" + str(int(key.split("+")[-1])) + "_" + str(num_frame) + '.tiff'
                else:
                    img['Image'] = "gtav_cid" + str(int(re.sub('[cid]', '', img['Image'].split('_')[1]))) + "_c" + str(int(key.split("+")[-1])) + "_" + str(num_frame+addNum) + '.tiff'
                
                # Check if image exists, if not, don't process the data;
                if not Path(join(in_directory, d['Image'])).is_file():
                    missingFiles += 1
                    print("Image %s not found!"%img["Image"])
                    continue

                # Transform 3D game coords into 2D image coords for bounding boxes
                for i, det in enumerate(img["Detections"]):
                    if det["Visibility"] == True:
                        pList = list()
                        pList.append(np.array([det["FURGame"]["X"],det["FURGame"]["Y"],det["FURGame"]["Z"]], dtype=np.float64))
                        pList.append(np.array([det["FULGame"]["X"],det["FULGame"]["Y"],det["FULGame"]["Z"]], dtype=np.float64))
                        pList.append(np.array([det["BULGame"]["X"],det["BULGame"]["Y"],det["BULGame"]["Z"]], dtype=np.float64))
                        pList.append(np.array([det["BURGame"]["X"],det["BURGame"]["Y"],det["BURGame"]["Z"]], dtype=np.float64))
                        pList.append(np.array([det["FLLGame"]["X"],det["FLLGame"]["Y"],det["FLLGame"]["Z"]], dtype=np.float64))
                        pList.append(np.array([det["BLLGame"]["X"],det["BLLGame"]["Y"],det["BLLGame"]["Z"]], dtype=np.float64))
                        pList.append(np.array([det["BLRGame"]["X"],det["BLRGame"]["Y"],det["BLRGame"]["Z"]], dtype=np.float64))
                        pList.append(np.array([det["FLRGame"]["X"],det["FLRGame"]["Y"],det["FLRGame"]["Z"]], dtype=np.float64))

                        BB3D = convertToCamCord(pList, img['Camrot'], img['Campos'], img['CamNearClip'], img['CamFOV'], img['ImageWidth'], img['ImageHeight'], img['UIwidth'], img['UIheight'])
                        BB2D = get2DBB(BB3D, img['ImageWidth'], img['ImageHeight'], img['UIwidth'], img['UIheight'])

                        det["FUR"] = {"X":BB3D[0][0], "Y": BB3D[0][1]}
                        det["FUL"] = {"X":BB3D[1][0], "Y": BB3D[1][1]}
                        det["BUL"] = {"X":BB3D[2][0], "Y": BB3D[2][1]}
                        det["BUR"] = {"X":BB3D[3][0], "Y": BB3D[3][1]}
                        det["FLL"] = {"X":BB3D[4][0], "Y": BB3D[4][1]}
                        det["BLL"] = {"X":BB3D[5][0], "Y": BB3D[5][1]}
                        det["BLR"] = {"X":BB3D[6][0], "Y": BB3D[6][1]}
                        det["FLR"] = {"X":BB3D[7][0], "Y": BB3D[7][1]}

                        det["BBmin"] = {"X": BB2D[0][0][0],"Y": BB2D[0][0][1]}
                        det["BBmax"] = {"X": BB2D[0][1][0], "Y": BB2D[0][1][1]}
                        det["Pos2D"] = {"X": int(BB2D[0][0][0]+(BB2D[0][1][0]-BB2D[0][0][0])/2), "Y": int(BB2D[0][0][1]+(BB2D[0][1][1]-BB2D[0][0][1])/2)}

                        img["Detections"][i] = det
                        allBB += 1

                        # Convert bounding boxes in Rectangle-object for simple occlusion checking
                        bbList.append(Rectangle(det["BBmin"]["X"], det["BBmin"]["Y"], det["BBmax"]["X"], det["BBmax"]["Y"]))

                # Calculate better 2D bounding boxes
                # Read stencil image
                im_s = cv2.imread(join(in_directory, img["Image"].split(".")[0] + stencil_s))
                # Mask array of cars set to 255, rest to 0
                try:
                    mask = bbt.maskOff(maskID, im_s)
                except:
                    errorImg += 1
                    print("[ERROR] in bbt.maskOff for image %s"%img["Image"])
                    continue

                # Stencil is in 8-bit int
                mask.astype(np.uint8)

                # Read depth image
                im_d = np.array(tifffile.imread(join(in_directory, img["Image"].split(".")[0] + depth_s)))
                # Convert float to int
                maxF = np.max(im_d)
                im_d = 1*im_d/maxF
                #im_d = im_d.astype(np.uint8)

                # Mask depth image based in stencil; remove non-car pixels
                im_d_masked = cv2.bitwise_and(im_d, im_d, mask=mask)

                # Optimize the bounding boxes
                for i, det in enumerate(img["Detections"]):
                    isOccluded = 0
                    if det["Visibility"] == True:
                        # Check if 2D bounding box is contained in another bounding box. This happens for example when cars overlap and the car behind is not visible, but its bounding box
                        # appears within the bounding box of the front car; this is just a simple quick fix
                        # https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles/25068722#25068722
                        b = Rectangle(det["BBmin"]["X"], det["BBmin"]["Y"], det["BBmax"]["X"], det["BBmax"]["Y"])
                        for r in bbList:
                            if b == r:
                                continue
                            # If rectangle is contained in another, it will create 8 rectangles; in that case, set visibility to false 
                            if len(list(r-b)) == 8:
                                isOccluded = 1
                                continue
                        
                        if isOccluded:
                            det["Visibility"] == False
                            img["Detections"][i] = det
                            continue

                        # Use depth and stencil buffer to get better 2D bounding boxes
                        # Get old bounding box from json as (minx, miny, width, height)
                        minx, miny, w, h = det["BBmin"]["X"], det["BBmin"]["Y"], det["BBmax"]["X"]-det["BBmin"]["X"], det["BBmax"]["Y"]-det["BBmin"]["Y"]

                        # Cut out mask and depth images based on the old bounding box for this car
                        cut_d_masked = im_d_masked[miny:miny+h, minx:minx+w]
                        cut_mask = mask[miny:miny+h, minx:minx+w]

                        # Check if array is all zero, if so, skip this
                        if not np.any(cut_d_masked):
                            img["Detections"][i] = det
                            continue

                        # Calculate contour and center of stencil blobs and draw bounding box
                        contours, centers, bb = bbt.findContours(cut_mask)
                        if not bb:
                            img["Detections"][i] = det
                            continue
                            
                        # Get bounding box based on stencil map
                        newBB = bbt.getStencilBB(bb)
                        # Create new mask with stencil blobs, since stencil image cannot be used due to data format (8-bit)
                        nmask = np.zeros((cut_mask.shape[0], cut_mask.shape[1]), np.uint8)
                        # Draw contours on new mask; those will be the boundaries for the flood fill
                        cv2.drawContours(nmask, contours, -1, (255, 255, 255), 1)
                        # Divide the bounding boxes of the stencil blobs and calculate for each new rectangle the centers, which will be added to the seed points later
                        divider = bbt.getDivider(w, h)
                        bbcenters = bbt.divideBB(bb, divider)
                        # Check if the new seed points are lying in the contours and add them to the list
                        acceptSeeds = bbt.centerInPoly(bbcenters, contours)

                        allSeeds = list()
                        # Add center of stencil blobs as seeds and also the centers of the divided bounding boxes
                        allSeeds.extend(centers)
                        allSeeds.extend(acceptSeeds)

                        if not allSeeds:
                            img["Detections"][i] = det
                            continue

                        # Do flood fill on the masked depth image with all the seeds
                        ffim, ffmask, rects = bbt.ff(nmask, cut_d_masked, allSeeds, lL, uL)

                        #print("BBold: (%d, %d, %d, %d)"%(minx, miny, w, h))
                        # Calculate IOU and use threshold to decide if new bounding box is accepted
                        niou = 0
                        if len(rects) is 1:
                            r = rects[0]
                            #print("BBnew: (%d, %d, %d, %d)"%(minx+r[0], miny+r[1], r[2], r[3]))
                            # bb1 = {'x1':minx, 'x2':minx+w, 'y1':miny, 'y2':miny+h} # Original 2D BB derived from 3D BB
                            bb1 = {'x1':minx+newBB[0], 'x2':minx+newBB[0]+newBB[2], 'y1':miny+newBB[1], 'y2':miny+newBB[1]+newBB[3]}
                            bb2 = {'x1':minx+r[0], 'x2':minx+r[0]+r[2], 'y1':miny+r[1], 'y2':miny+r[1]+r[3]}
                            niou = bbt.get_iou(bb1, bb2)
                            #print("IOU: %f"%niou)
                            #print("Adding better 2D bounding box!")
                            det["BBminNew"] = {"X": minx+r[0],"Y": miny+r[1]}
                            det["BBmaxNew"] = {"X": minx+r[0]+r[2], "Y": miny+r[1]+r[3]}
                            det["Pos2DNew"] = {"X": int(minx+r[0]+(r[2]/2)), "Y": int(miny+r[1]+(r[3]/2))}
                            det["IOU"] = niou
                            optimizedBB += 1
                        elif not not rects:
                            # Get biggest final bounding box
                            finalBB = bbt.getStencilBB(rects)
                            bb1 = {'x1':minx+newBB[0], 'x2':minx+newBB[0]+newBB[2], 'y1':miny+newBB[1], 'y2':miny+newBB[1]+newBB[3]}
                            bb2 = {'x1':minx+finalBB[0], 'x2':minx+finalBB[0]+finalBB[2], 'y1':miny+finalBB[1], 'y2':miny+finalBB[1]+finalBB[3]}
                            niou = bbt.get_iou(bb1, bb2)
                            #print("IOU: %f"%niou)
                            #print("Adding better 2D bounding box!")
                            det["BBminNew"] = {"X": minx+finalBB[0],"Y": miny+finalBB[1]}
                            det["BBmaxNew"] = {"X": minx+finalBB[0]+finalBB[2], "Y": miny+finalBB[1]+finalBB[3]}
                            det["Pos2DNew"] = {"X": int(minx+finalBB[0]+(finalBB[2]/2)), "Y": int(miny+finalBB[1]+(finalBB[3]/2))}
                            det["IOU"] = niou
                            optimizedBB += 1
                        
                        img["Detections"][i] = det
                total.append(img)
            else:
                print("[WARNING] No offsetting between annotation and image found for: \n%s"%join(in_directory, img['Image']))

    border_msg("Saving data to json data_boxes.json")
    with open('data_boxes.json', 'w') as outfile:
        json.dump(total, outfile, indent=2)
        #outfile.write('\n')

    border_msg("Some statistics:")
    print("\nNumber of all bounding boxes: %d"%allBB)
    print("Number of discarded images: %d"%missingFiles)
    print("Number of errornous images: %d"%errorImg)
    print("Number of optimized bounding boxes: %d"%optimizedBB)

def border_msg(msg):
    row = len(msg)
    h = ''.join(['+'] + ['-' *row] + ['+'])
    result= h + '\n'"|"+msg+"|"'\n' + h
    print(result)

def filterSmall(fs):
    data = []
    print("Preparing data ...")
    with open("data_boxes.json") as reader:
        read = json.load(reader)
        for line in read:
            data.append(line)
    cnt = 0
    missing_cnt = 0
    print("Counting small bounding boxes ..")
    for img in data:
        for det in img["Detections"]:
            if det["Type"] == "car" and det["Visibility"] == True:
                if "BBmaxNew" in det:
                    if (det["BBmaxNew"]["Y"] - det["BBminNew"]["Y"])/ img["ImageHeight"] < fs:
                        cnt += 1
                elif "BBmax" in det:
                    if (det["BBmax"]["Y"] - det["BBmin"]["Y"])/ img["ImageHeight"]  < fs:
                        cnt += 1
                else:
                    missing_cnt += 1

    print("In total, %d bounding boxes are smaller then %f"%(cnt, fs))
    print("In total, %d detections were without 2D bounding boxes."%missing_cnt)

def test():
    data = []
    print("Preparing data ...")
    with open("data_boxes.json") as reader:
        read = json.load(reader)
        for line in read:
            data.append(line)

    for img in data:
        imname = 'D:\\Devel\\GTAVisionExport\\managed\\Data\\' + img["Image"]
        print(imname)
        imgPil = Image.open(imname)
        # cv2.cvtColor macht am FZI Rechner Probleme!
        im = cv2.cvtColor(np.array(imgPil), cv2.COLOR_BGR2RGB)
        image = np.array(im)

        for det in img["Detections"]:
            if det["Type"] == "car" and det["Visibility"] == True:
                if "BBmaxNew" in det:
                    meanCol = cv2.mean(image[det["BBminNew"]["Y"]:det["BBmaxNew"]["Y"], det["BBminNew"]["X"]:det["BBmaxNew"]["X"]])                   
                    image[det["BBminNew"]["Y"]:det["BBmaxNew"]["Y"], det["BBminNew"]["X"]:det["BBmaxNew"]["X"]] = meanCol[:-1]

                elif "BBmax" in det:
                    meanCol = cv2.mean(image[det["BBmin"]["Y"]:det["BBmax"]["Y"], det["BBmin"]["X"]:det["BBmax"]["X"]]) 
                    image[det["BBmin"]["Y"]:det["BBmax"]["Y"], det["BBmin"]["X"]:det["BBmax"]["X"]] = meanCol[:-1]
            
        cv2.imshow("Test",image)

        key = cv2.waitKey(0)
        if key == 27: # escape
            cv2.destroyAllWindows()
            break
                


if __name__ == '__main__':
    # python visualizeGTA.py --json 1
    # python visualizeGTA.py --json 1 --mask 2 --lL 0.00099 --ul 0.009
    # python visualizeGTA.py --plot 0
    # python visualizeGTA.py --plot 1 --iou 0.1 --minw 25 --minh 10
    # --mode 0: show 2D & 3D bounding boxes
    # --mode 1: show old (in green) and new (in red) 2D bounding boxes
    # --mode 2: show 2D bounding boxes
    # --mode 3: show 3D bounding boxes

    parser = argparse.ArgumentParser()
    # Plot image with bbs or plot image with stencil and depth
    parser.add_argument('--plot', default=None, type=int)
    # Chooses which bbs are plotted, 2D, 3D, both, old & new 2D
    parser.add_argument('--mode', default=0, type=int)
    # Command to create json
    parser.add_argument('--json', default=None, type=int)
    # Offseting between image and annotation, since data has one frame lag
    parser.add_argument('--off', default=1, type=int)
    # ID of the stencil mask, which should be used for bb optimization
    parser.add_argument('--mask', default=2, type=int)
    # IOU threshold for accepting optimized bb, 0.5 is good
    parser.add_argument('--iou', default=0.5, type=float)
    # Lower limit parameter for flood fill algo
    parser.add_argument('--lL', default=0.005, type=float)
    # Upper limit parameter for flood fill algo
    parser.add_argument('--uL', default=0.05, type=float)
    # Minimum old bounding box width to be selected for bb optimization
    parser.add_argument('--minw', default=25, type=int)
    # Minimum old bounding box height to be selected for bb optimization
    parser.add_argument('--minh', default=10, type=int)
    # Filter to count bounding boxes smaller then a fraction of relative image height
    parser.add_argument('--filter', default=None, type=float)
    # Reserved for frequently changing test functions
    parser.add_argument('--test', default=None, type=float)
    args = parser.parse_args()

    if args.filter: filterSmall(args.filter)
    if args.test: test()

    if args.json is not None:
        createJson(args.off, args.mask, args.lL, args.uL)
    # Plot image with bbs
    elif args.plot == 1: plotCV(args.mode, args.iou, args.minw, args.minh)
    # Plot color, depth and stencil image
    elif args.plot == 0: main()
