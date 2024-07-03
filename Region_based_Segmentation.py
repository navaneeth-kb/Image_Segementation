import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color

img=plt.imread(r"C:\Users\navan\Downloads\1.jpeg")
plt.imshow(img)
print(img.shape)
plt.show()

gray = color.rgb2gray(img)
plt.imshow(gray, cmap='gray')
plt.show()

gray = np.array(gray)
gray_mean = gray.mean()

# Flatten the array for processing
gray_flattened = gray.reshape(-1)
print(gray_flattened)

# Apply the thresholding
for i in range(gray_flattened.shape[0]):
    if gray_flattened[i] < gray_mean:
        gray_flattened[i] = 0
    else:
        gray_flattened[i] = 1

# Reshape back to the original image shape
binary_img = gray_flattened.reshape(gray.shape)

# Display the binarized image
plt.imshow(binary_img, cmap='gray')
plt.axis('off')  # Hide the axis
plt.show()

'''
One simple way to segment different objects could be to use their pixel values. An important point to note – the pixel values will be different for the objects and the image’s background if there’s a sharp contrast between them.
In this case, we can set a threshold value. The pixel values falling below or above that threshold can be classified accordingly (as objects or backgrounds). This technique is known as Threshold Segmentation.
'''
