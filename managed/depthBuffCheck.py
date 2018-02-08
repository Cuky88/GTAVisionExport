import numpy as np 
import cv2
from PIL import Image

width = 1280
height = int(720/2)
rowPitch = 10240 # read from native 

depthBuf = open('depth.raw', 'rb')
depth = np.fromfile(depthBuf).reshape(height, width)
#d = np.frombuffer(depthBuf)
# print(len(depth)) # = 460.800 
#d = np.empty([height, width]).reshape(height, width)
depth.astype(dtype="float32")


img = Image.fromarray(depth, 'RGB')
img.save('test.png')
img.show()



# Display the OpenCV image using inbuilt methods.
cv2.imshow('Demo Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()