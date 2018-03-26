import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageFile, ImageDraw, ImageFont
from skimage import io
from matplotlib import cm
from matplotlib.path import Path
import matplotlib.patches as patches
import tifffile
from os import listdir, makedirs
from os.path import isfile, join
import cv2

in_directory_depth = "Data/GTAV/png/depth_"

images = [f for f in listdir(in_directory_depth) if isfile(join(in_directory_depth, f))]

for img in images:
    #img = Image.open('gtav_cid0_c258_14603-depth.png')
    im = cv2.imread(join(in_directory_depth, img),  cv2.IMREAD_UNCHANGED)
    #img=mpimg.imread(('gtav_cid0_c258_14603-depth.png')

    bidx = im>75
    im[bidx] = 0

    cv2.imwrite(join(in_directory_depth, img), im)
