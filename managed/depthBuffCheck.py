import numpy as np 
import cv2
from PIL import Image

width = 1280
height = 720
rowPitch = 10240 # read from native 

depthBuf = open('depth.raw', 'rb')
depth = np.fromfile(depthBuf)
#d = np.frombuffer(depthBuf)
# print(len(depth)) # = 460.800 
d = np.empty([height, width])
for x in range(width):
    print("x: %d"%x)
    for y in range(height):
        print("y: %d"%y)
        d[y][x] = depth[y + x * 8]

print(d)

img = Image.fromarray(d, 'RGB')
img.save('test.png')
img.show()

''' for (int x = 0; x < 1280; ++x)
	{
		for (int y = 0; y < 720; ++y)
		{
			const float* src_f = (const float*)((const char*)src_map.pData + src_map.RowPitch*y + (x * 8));
			unsigned char* dst_p = &dst[src_desc.Width * 4 * y + (x * 4)];
			unsigned char* stencil_p = &stencil[src_desc.Width * y + x];
			memmove(dst_p, src_f, 4);
			memmove(stencil_p, src_f + 1, 1);
		}
	} '''

# Display the OpenCV image using inbuilt methods.
cv2.imshow('Demo Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()