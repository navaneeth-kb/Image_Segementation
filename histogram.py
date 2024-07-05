import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color, exposure, filters
from scipy import ndimage

data_dir = r'C:\Users\navan\Downloads\archive\DRIVE'
train_images_dir = os.path.join(data_dir, r'training\images')

for img_file in os.listdir(train_images_dir):
    img_path = os.path.join(train_images_dir, img_file)
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title('Original Image')
    plt.show()

    gray = color.rgb2gray(img)

    # Apply histogram equalization
    gray_eq = exposure.equalize_hist(gray)

    plt.imshow(gray_eq, cmap='gray')
    plt.title('Histogram Equalized Image')
    plt.show()

    # Calculate the histogram of the equalized grayscale image
    hist, bin_edges = np.histogram(gray_eq.flatten(), bins=256, range=[0, 1])
    
    # Plot the histogram
    plt.plot(bin_edges[0:-1], hist)
    plt.title('Histogram')
    plt.show()
    
    # Find the threshold based on the histogram analysis
    threshold_value = filters.threshold_otsu(gray_eq)
    
    # Apply thresholding
    binary_out = gray_eq > threshold_value

    plt.imshow(binary_out, cmap='gray')
    plt.title('Binary Image after Histogram Segmentation')
    plt.show()
