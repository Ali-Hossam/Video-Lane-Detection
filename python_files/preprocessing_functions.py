import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray,rgb2hsv
from skimage.filters import median, gaussian
from skimage import morphology
import cv2

def histogram_equalization(img):
    # Load image using io.imread()
#     img = io.imread(image_path)

    # Compute histogram
    hist, bins = np.histogram(img.flatten(), bins=256)

    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = (cdf * hist.max()) / cdf.max()

    # Perform histogram equalization
    equalized_img = np.interp(img.flatten(), bins[:-1], cdf_normalized).reshape(img.shape)

    # ** normalize image pixel values
    min_value = np.min(equalized_img)
    max_value = np.max(equalized_img)
    # print(equalized_img, min_value, max_value)
    image_normalized = (equalized_img - min_value) / (max_value - min_value)
    
    return image_normalized

def plot_images(img1, img2, label1, label2):
    """Plots two images side by side."""
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the first image on the first subplot
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(label1)

    # Plot the second image on the second subplot
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(label2)

    # Show the plot
    plt.tight_layout()
    plt.show()

def ROI(img):
    H, W, _ = img.shape
    cropped_img = img[H//2 : H, :]
    return cropped_img

def smooth(image):
    """Applies median filter to smooth image."""
    gray_image = rgb2gray(image)
    f_image = median(gray_image)
    return f_image

def gray_to_binary(img):
    """converts a gray scaled image to binary."""
    gray_image_8bit = cv2.convertScaleAbs(img)
    _, binary_image = cv2.threshold(gray_image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def apply_errosion(img, element_size):
    element = morphology.rectangle(element_size, element_size+5)
    opened_img = morphology.binary_erosion(img, element)
    return opened_img
