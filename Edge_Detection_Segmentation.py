import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color
from scipy import ndimage

img=plt.imread(r"C:\Users\navan\Downloads\1.jpeg")

plt.imshow(img)
plt.show()

gray = color.rgb2gray(img)
plt.imshow(gray, cmap='gray')
plt.show()

kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
plt.imshow(out_l, cmap='gray')
plt.show()

'''
What divides two objects in an image? An edge is always between two adjacent regions with different grayscale values (pixel values). The edges can be considered as the discontinuous local features of an image.
We can use this discontinuity to detect edges and hence define a boundary of the object. This helps us detect the shapes of multiple objects in a given image. Now, the question is, how can we detect these edges? This is where we can make use of filters and convolutions.
'''
