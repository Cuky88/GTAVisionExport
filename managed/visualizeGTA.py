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

depths = {}
stencils = {}
in_directory = 'Data\\'
out_directory = 'Data\\img2\\'

objlist = [f for f in listdir(in_directory) if isfile(join(in_directory, f)) and f.split(".")[-1]=="json"]
data = []

for obj in objlist:
    print(join(in_directory, obj))
    with open(join(in_directory, obj)) as file:
        for line in file:
            data.append(json.loads(line))


def rotate(p, theta):
    # Rotation order: Z Y X
    X = np.cos(theta[2]) * (np.cos(theta[1]) * p[0] + np.sin(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) -  np.sin(theta[2]) * (np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    #X = np.cos(theta[2]) * ((np.cos(theta[1]) * p[0] + np.sin(theta[1]) * ((np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) - np.sin(theta[2]) * ((np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    Y = np.sin(theta[2]) * (np.cos(theta[1]) * p[0] + np.sin(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) + np.cos(theta[2]) * (np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    #Y = np.sin(theta[2]) * (np.cos(theta[1]) * p[0] + np.sin(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) + np.cos(theta[2]) * (np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    Z = -np.sin(theta[1]) * p[0] + np.cos(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])
    #Z = -np.sin(theta[1]) * p[0] + np.cos(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])
    
    #print(X, Y, Z)
    return np.array([X,Y,Z])


def convertToCamCord(pList, Camrot, Campos, camNearClip, camFOV, imgw, imgh, uiw, uih):
    pListProj = list()
    pListBB = list()
    for p in pList:
        # camera rotation 
        theta = (np.pi / 180) * np.array([Camrot["X"],Camrot["Y"], Camrot["Z"]])
        #print("Theta: %s"%str(theta))

        # camera direction, at 0 rotation the camera looks down the postive Y axis --> WorldNorth schaut somit immer in Cam-Richtung
        camDir = rotate(np.array([0,1,0]), theta)
        #print("camDir: %s"%str(camDir))

        # camera position  == eigentlich Bildebene rotiert in Blickrichtung, minimal vor der Cam zu sehen
        c = np.array([Campos["X"],Campos["Y"], Campos["Z"]]) + camNearClip * camDir
        #print("c: %s"%str(c))

        # viewer position == Cam-Pos rotiert in Blickrichtung; minimal hinter c!
        e = -camNearClip * camDir
        #print("e: %s"%str(e))

        viewWindowHeight = 2 * camNearClip * np.tan((camFOV / 2) * (np.pi / 180))
        viewWindowWidth = (imgw*1. / imgh*1.) * viewWindowHeight;
        #print("viewWindowHeight: %s"%str(viewWindowHeight))
        #print("viewWindowWidth: %s"%str(viewWindowWidth))
        
        camUp = rotate(np.array([0,0,1]), theta)
        #print("camUp: %s"%str(camUp))
        
        camEast = rotate(np.array([1,0,0]), theta)
        #print("camEast: %s"%str(camEast))

        # Distanz zwischen Punkt und Bildebene
        delete = p - c
        #print("delete: %s"%str(delete))
        
        viewerDist = delete - e
        #print("viewerDist: %s"%str(viewerDist))
        
        # Vector3 viewerDistNorm = viewerDist * (1 / viewerDist.Length());
        viewerDistNorm = viewerDist/np.linalg.norm(viewerDist)
        #print("viewerDistNorm: %s"%str(viewerDistNorm))
        
        dot = np.dot(camDir, viewerDistNorm)
        ang = np.arccos(dot)
        #print("dot: %s"%str(dot))
        #print("ang: %s"%str(ang))
        
        # Senkrechter Abstand zur Bildebene
        viewPlaneDist = camNearClip / np.cos(ang)
        #print("viewPlaneDist: %s"%str(viewPlaneDist))
        
        # Punkt auf der Bildebene
        viewPlanePoint1 = viewPlaneDist * viewerDistNorm + e
        #print("viewPlanePoint: %s"%str(viewPlanePoint1))

        # move origin to upper left 
        newOrigin = c + (viewWindowHeight / 2) * camUp - (viewWindowWidth / 2) * camEast
        viewPlanePoint = (viewPlanePoint1 + c) - newOrigin
        #print("newOrigin: %s"%str(newOrigin))
        #print("viewPlanePoint: %s"%str(viewPlanePoint))

        viewPlaneX = np.dot(viewPlanePoint, camEast) / np.dot(camEast, camEast)
        viewPlaneZ = np.dot(viewPlanePoint, camUp) / np.dot(camUp, camUp)
        #print("viewPlaneX: %s"%str(viewPlaneX))
        #print("viewPlaneZ: %s"%str(viewPlaneZ))

        screenX = viewPlaneX / viewWindowWidth * uiw
        screenY = -viewPlaneZ / viewWindowHeight * uih
        #print("screenX: %s"%str(screenX))
        #print("screenY: %s"%str(screenY))
        
        Xscale = float(imgw) / (1.0 * uiw) * screenX
        Yscale = float(imgh) / (1.0 * uih) *screenY
        #print("Xscale: %s"%str(Xscale))
        #print("Yscale: %s"%str(Yscale))

        pListProj.append((int(Xscale), int(Yscale)))
    
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    
    i = 0
    pOut = list()
    for p in pListProj:
        if p[0] < 0 or p[1] < 0:
            i += 1
            pOut.append(p)
            continue
        
        if p[0] < minX: minX = p[0]
        if p[0] > maxX: maxX = p[0]
        if p[1] < minY: minY = p[1]
        if p[1] > maxY: maxY = p[1]
    
    if minX < 0: minX = 0
    if minY < 0: minY = 0

    for o in pOut:
        if o[0] < 0:
            minX = 0
        elif o[0] > uiw:
            maxX = uiw
        elif o[1] < 0:
            minY = 0
        elif o[1] > uih:
            maxY = uih

    if maxX > uiw: maxX = uiw
    if maxY > uih: maxY = uih
    
    pListBB.append([(int(minX), int(minY)), (int(maxX), int(maxY))])
    return pListProj, pListBB

def bbox_from_string(name):
    bbox2 = []
    bbox3 = []
    for d in data:
        if d['Image'] == name:
            print( d['Campos'])
            for i,p in enumerate(d['Detections']):
                if p["Type"] == "car" and p["Visibility"]:
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

def show_bounding_boxes(name, size, ax):
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
    return arr * 4


def main():
    files = [f for f in listdir(in_directory) if isfile(join(in_directory, f)) and f.split(".")[-1]=="tiff" and len(f.split("-"))==1]
    #files = ["gtav_0.tiff", "gtav_1.tiff", "gtav_2.tiff", "gtav_4.tiff"]
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for name in files:
        print(name)
        #img = cv2.imread(os.path.join(in_directory, name), 1)
        #size = (1280,720)

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
        ax1.imshow(im)
        # ax2.set_title('f')
        ax2.imshow(ids_to_greyscale(load_stencil_ids(name)), cmap='gray')
        ax3.set_title('ids')
        # ax3.imshow(load_stencil_ids(name), cmap='gray')
        ax3.imshow(ids_to_greyscale(load_stencil_ids(name)), cmap='plasma')
        ax4.set_title('depth')
        ax4.imshow(load_depth(name), cmap='gray')
        
        # Der funzt einigermassen gut; Allg. Punktezeichnen bei Matplotlib ist nicht sonderlich gut
        #show_bounding_boxes(name, size, ax1)

        plt.axis('off')
        plt.draw()
        plt.show()

    plt.show()

def main2():
    files = [f for f in listdir(in_directory) if isfile(join(in_directory, f)) and f.split(".")[-1]=="tiff" and len(f.split("-"))==1]
    #files = ["gtav_3.tiff", "gtav_1.tiff", "gtav_2.tiff", "gtav_4.tiff"]
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for name in files:       
        bbox2 = []
        for d in data:
            if d['Image'] == name and int(name.split("_")[-1][:-5])+1 < len(files):
                splitname = name.split("_")
                imname = 'D:\\Devel\\GTAVisionExport\\managed\\Data\\' + splitname[0] + "_" + str(int(splitname[-1][:-5])+1) + ".tiff"
                print(imname)
                imgPil = Image.open(imname)

                # cv2.cvtColor macht am FZI Rechner Probleme!
                #img = cv2.cvtColor(np.array(imgPil), cv2.COLOR_BGR2RGB)
                img = np.array(imgPil)

                if img == None: print("Image %s not found!"%('D:\\Devel\\GTAVisionExport\\managed\\Data\\' + name)); break

                for i,p in enumerate(d['Detections']):
                    if p["Type"] == "car" and p["Visibility"]:
                        
                        #print((p["FUR"]["X"],p["FUR"]["Y"]), (p["FUL"]["X"],p["FUL"]["Y"]), (p["BUL"]["X"],p["BUL"]["Y"]), (p["BUR"]["X"],p["BUR"]["Y"]), 
                        #(p["FLL"]["X"],p["FLL"]["Y"]), (p["BLL"]["X"],p["BLL"]["Y"]), (p["BLR"]["X"],p["BLR"]["Y"]), (p["FLR"]["X"],p["FLR"]["Y"]))
                        
                        pList = list()
                        pList.append(np.array([p["FURGame"]["X"],p["FURGame"]["Y"],p["FURGame"]["Z"]]))
                        pList.append(np.array([p["FULGame"]["X"],p["FULGame"]["Y"],p["FULGame"]["Z"]]))
                        pList.append(np.array([p["BULGame"]["X"],p["BULGame"]["Y"],p["BULGame"]["Z"]]))
                        pList.append(np.array([p["BURGame"]["X"],p["BURGame"]["Y"],p["BURGame"]["Z"]]))
                        pList.append(np.array([p["FLLGame"]["X"],p["FLLGame"]["Y"],p["FLLGame"]["Z"]]))
                        pList.append(np.array([p["BLLGame"]["X"],p["BLLGame"]["Y"],p["BLLGame"]["Z"]]))
                        pList.append(np.array([p["BLRGame"]["X"],p["BLRGame"]["Y"],p["BLRGame"]["Z"]]))
                        pList.append(np.array([p["FLRGame"]["X"],p["FLRGame"]["Y"],p["FLRGame"]["Z"]]))

                        BB3D, BB2D = convertToCamCord(pList, d['Camrot'], d['Campos'], d['CamNearClip'], d['CamFOV'], d['ImageWidth'], d['ImageHeight'], d['UIwidth'], d['UIheight'])
                        
                        print("\n2D Bounding Boxes:")
                        print(BB2D)
                        print("\n3D Bounding Boxes:")
                        print(BB3D)
                        print("-------------------")

                        if int(sys.argv[1]) == 2:
                            for bb in BB2D:
                                img = cv2.rectangle(img, bb[0], bb[1], (255, 255, 0), 1)

                        elif int(sys.argv[1]) == 3:
                            img = cv2.line(img, BB3D[0], BB3D[1], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[1], BB3D[2], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[2], BB3D[3], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[3], BB3D[0], (255, 0, 255), 1)

                            img = cv2.line(img, BB3D[4], BB3D[5], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[5], BB3D[6], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[6], BB3D[7], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[7], BB3D[4], (255, 0, 255), 1)

                            img = cv2.line(img, BB3D[0], BB3D[7], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[1], BB3D[4], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[2], BB3D[5], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[3], BB3D[6], (255, 0, 255), 1)

                        elif int(sys.argv[1]) == 4:
                            for bb in BB2D:
                                img = cv2.rectangle(img, bb[0], bb[1], (255, 255, 0), 1)

                            img = cv2.line(img, BB3D[0], BB3D[1], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[1], BB3D[2], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[2], BB3D[3], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[3], BB3D[0], (255, 0, 255), 1)

                            img = cv2.line(img, BB3D[4], BB3D[5], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[5], BB3D[6], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[6], BB3D[7], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[7], BB3D[4], (255, 0, 255), 1)

                            img = cv2.line(img, BB3D[0], BB3D[7], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[1], BB3D[4], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[2], BB3D[5], (255, 0, 255), 1)
                            img = cv2.line(img, BB3D[3], BB3D[6], (255, 0, 255), 1)
                        
                        #img = cv2.line(img, (int(p["FUR"]["X"]),int(p["FUR"]["Y"])), (int(p["FUL"]["X"]),int(p["FUL"]["Y"])), (255, 0, 255), 1)
                        #img = cv2.line(img, (int(p["FUL"]["X"]),int(p["FUL"]["Y"])), (int(p["BUL"]["X"]),int(p["BUL"]["Y"])), (255, 0, 255), 1)
                        #img = cv2.line(img, (int(p["BUL"]["X"]),int(p["BUL"]["Y"])), (int(p["BUR"]["X"]),int(p["BUR"]["Y"])), (255, 0, 255), 1)
                        #img = cv2.line(img, (int(p["BUR"]["X"]),int(p["BUR"]["Y"])), (int(p["FUR"]["X"]),int(p["FUR"]["Y"])), (255, 0, 255), 1)

                        #img = cv2.line(img, (int(p["FLR"]["X"]),int(p["FLR"]["Y"])), (int(p["FLL"]["X"]),int(p["FLL"]["Y"])), (255, 0, 255), 1)
                        #img = cv2.line(img, (int(p["FLL"]["X"]),int(p["FLL"]["Y"])), (int(p["BLL"]["X"]),int(p["BLL"]["Y"])), (255, 0, 255), 1)
                        #img = cv2.line(img, (int(p["BLL"]["X"]),int(p["BLL"]["Y"])), (int(p["BLR"]["X"]),int(p["BLR"]["Y"])), (255, 0, 255), 1)
                        #img = cv2.line(img, (int(p["BLR"]["X"]),int(p["BLR"]["Y"])), (int(p["FLR"]["X"]),int(p["FLR"]["Y"])), (255, 0, 255), 1)

                        #img = cv2.line(img, (int(p["FUR"]["X"]),int(p["FUR"]["Y"])), (int(p["FLR"]["X"]),int(p["FLR"]["Y"])), (255, 0, 255), 1)
                        #img = cv2.line(img, (int(p["FUL"]["X"]),int(p["FUL"]["Y"])), (int(p["FLL"]["X"]),int(p["FLL"]["Y"])), (255, 0, 255), 1)
                        #img = cv2.line(img, (int(p["BUL"]["X"]),int(p["BUL"]["Y"])), (int(p["BLL"]["X"]),int(p["BLL"]["Y"])), (255, 0, 255), 1)
                        #img = cv2.line(img, (int(p["BUR"]["X"]),int(p["BUR"]["Y"])), (int(p["BLR"]["X"]),int(p["BLR"]["Y"])), (255, 0, 255), 1)


                cv2.imshow(name,img)

                key = cv2.waitKey(0)
                cv2.destroyAllWindows()

                if key == 27: # escape
                    break

if __name__ == '__main__':
    if len(sys.argv) > 1: main2()
    else: main()