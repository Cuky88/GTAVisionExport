from PIL import Image, ImageFile, TiffImagePlugin, ImageSequence
import matplotlib.pyplot as pp
import matplotlib.cm as cm
import numpy as np

Image.DEBUG = True
TiffImagePlugin.WRITE_LIBTIFF = True


im = Image.open('info23.tiff')

for frame in ImageSequence.Iterator(im):
    frame.load()
    frame.save("iter%d.tif" % im.tell())
	
pp.imshow(im, vmin=0, vmax=255, cmap=cm.Spectral)
pp.show()