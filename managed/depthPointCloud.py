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
from scipy import misc
from os import listdir, makedirs
from os.path import isfile, join
import json
import cv2
import shutil
import glob
import re
from subprocess import call
import math
import itertools
from pathlib import Path
from tqdm import tqdm


in_directory = 'Data\\'
out_directory = 'Data\\img2\\'
DEBUG_TRANS = False

def rotate(p, theta):
    # Rotation order: Z Y X
    X = np.cos(theta[2]) * (np.cos(theta[1]) * p[0] + np.sin(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) -  np.sin(theta[2]) * (np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    #X = np.cos(theta[2]) * ((np.cos(theta[1]) * p[0] + np.sin(theta[1]) * ((np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) - np.sin(theta[2]) * ((np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    Y = np.sin(theta[2]) * (np.cos(theta[1]) * p[0] + np.sin(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) + np.cos(theta[2]) * (np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    #Y = np.sin(theta[2]) * (np.cos(theta[1]) * p[0] + np.sin(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])) + np.cos(theta[2]) * (np.cos(theta[0]) * p[1] - np.sin(theta[0]) * p[2])
    Z = -np.sin(theta[1]) * p[0] + np.cos(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])
    #Z = -np.sin(theta[1]) * p[0] + np.cos(theta[1]) * (np.sin(theta[0]) * p[1] + np.cos(theta[0]) * p[2])

    return np.array([X,Y,Z])

data = []
with open("data_boxes.json") as reader:
    read = json.load(reader)
    for line in read:
        # Calculate View Matrix manually
        # camera rotation 
        theta = (np.pi / 180) * np.array([line['Camrot']["X"],line['Camrot']["Y"], line['Camrot']["Z"]], dtype=np.float64)

        # camera direction, at 0 rotation the camera looks down the postive Y axis --> WorldNorth schaut somit immer in Cam-Richtung
        camNorth = np.append(rotate(np.array([0,1,0]), theta), 0)
        camUp = np.append(rotate(np.array([0,0,1]), theta), 0)
        camEast = np.append(rotate(np.array([1,0,0]), theta), 0)

        viewMat = np.zeros((4, 4))
        viewMat[0,:] = camEast
        viewMat[1,:] = camUp
        viewMat[2,:] = -1.0*camNorth

        cx = np.dot(viewMat[0,:],-1.0*np.array([[line['Campos']["X"]], [line['Campos']["Y"]], [line['Campos']["Z"]], [0]]))
        cy = np.dot(viewMat[1,:],-1.0*np.array([[line['Campos']["X"]], [line['Campos']["Y"]], [line['Campos']["Z"]], [0]]))
        cz = np.dot(viewMat[2,:],-1.0*np.array([[line['Campos']["X"]], [line['Campos']["Y"]], [line['Campos']["Z"]], [0]]))

        viewMat[:,3] = np.array([cx, cy, cz, 1])

        line['VMatrix'] = viewMat

        # Calculate Projection Matrix manually
        # No need to calculate projection matrix; it is only based on FOV and Near/Far clip and not camera position or rotation. Besides that,
        # near and far clip aren't 0.15 & 800 in the prjection matrix from game; they ar 0.15 & 0.1499 - WHY?!

        # If calculating projection matrix myself, I get:
        #data['PMatrix'] = np.transpose(np.array([1.206285105340096, 0, 0, 0, 0, 2.144507022049367, 0, 0.0, 0, 0, -1.000375070325686066137400762643, -1.0, 0, 0, -0.30005626054885290992061011439645, 0.0]).reshape((4, 4)))

        # projMat = np.zeros((4, 4))
        # projMat[1,1] = 1.0 / np.tan((line['CamFOV']/2) * (np.pi / 180))
        # projMat[0,0] = (1.0 / np.tan((line['CamFOV']/2) * (np.pi / 180))) / (line['ImageWidth']/line['ImageHeight'])
        # projMat[2,2] = -1*((-1*line['CamFarClip']+(-1)*line["CamNearClip"]) / (-1*line['CamFarClip']-(-1)*line["CamNearClip"]))
        # projMat[3,2] = -1.0
        # projMat[2,3] = -1*((2* -1*line['CamFarClip']*(-1)*line["CamNearClip"]) / (-1*line['CamFarClip']-(-1)*line["CamNearClip"]))
        # line['PMatrix'] = projMat

        line['PMatrix'] = np.transpose(np.array(line['PMatrix']['Values']).reshape((4, 4)))

        data.append(line)

depths = {}
def load_depth(name):
    if name not in depths:
        tiff_depth = tifffile.imread(os.path.join(in_directory, name.split(".")[0] + '-depth.tiff'))
        depths[name.split(".")[0] + '-depth.tiff'] = tiff_depth
    return depths[name.split(".")[0] + '-depth.tiff']

def pixel_to_normalized(pixel, size):
    p_y, p_x = pixel
    s_y, s_x = size
    return ((2/s_y)*p_y - 1, (2/s_x)*p_x - 1)

def prepare_points(res):
    width = res['ImageWidth']
    height = res['ImageHeight']
    x_range = range(0, width)
    y_range = range(0, height)
    points = np.transpose([np.tile(y_range, len(x_range)), np.repeat(x_range, len(y_range))])
    return points

def points_to_homo(points, res, name):
    width = res['ImageWidth']
    height = res['ImageHeight']
    size = (height, width)
    depth = load_depth(name)
    proj_matrix = res['PMatrix']
    max_depth = res['CamFarClip']
    #max_depth = 60 # just for testing
    vec = proj_matrix @ np.array([[1], [1], [-max_depth], [1]])
    #print(vec)
    vec /= vec[3]
    #treshold = vec[2]

    vecs = np.zeros((4, points.shape[0]))
    #vecs = np.zeros((4, len(np.where(depth[points[:, 0], points[:, 1]] > treshold)[0])))  # this one is used when ommiting 0 depth (point behind the far clip)
    print("vecs.shape")
    print(vecs.shape)
    i = 0
    arr = points
    for y, x in arr:
        #if depth[(y, x)] <= treshold:
        #    continue
        ndc = pixel_to_normalized((y, x), size)
        vec = [ndc[1], -ndc[0], depth[(y, x)], 1]
        vec = np.array(vec)
        vecs[:, i] = vec
        i += 1

    return vecs

def to_view(vecs, res):
    proj_matrix = res['PMatrix']
    vecs_p = np.linalg.inv(proj_matrix) @ vecs
    vecs_p /= vecs_p[3, :]
    return vecs_p

def to_world(vecs_p, res):
    view_matrix = res['VMatrix']
    vecs_p = np.linalg.inv(view_matrix) @ vecs_p
    vecs_p /= vecs_p[3, :]
    return vecs_p

def save_csv(vecs_p, name):
    a = np.asarray(vecs_p[0:3, :].T)
    np.savetxt("points-{}.csv".format(name), a, delimiter=",")

def getPNG(img, vecs_p, data, name):
    start = 0
    end = data['ImageHeight']
    for x in np.arange(data['ImageWidth']):
        img[:,x] = np.array(vecs_p[2, start:end])
        start += data['ImageHeight']
        end += data['ImageHeight']
    
    img = np.nan_to_num(img)
    cv2.imwrite(name+"-depth.png", img)

cnt = 0
for d in data:
    if cnt < 6:
        name = d['Image'].split(".")[0]
        points = prepare_points(d)
        vecs = points_to_homo(points, d, name)
        print('image {} points prepared'.format(name))
        vecs_p = to_view(vecs, d)
        vecs_p = to_world(vecs_p, d)
        print('image {} projected'.format(name))
        save_csv(vecs_p, name)
        print('image {} processed'.format(name))
        getPNG(np.zeros((d['ImageHeight'], d['ImageWidth'])), vecs_p, d, name)
        print('deapth map created')
    else:
        break
    cnt += 1