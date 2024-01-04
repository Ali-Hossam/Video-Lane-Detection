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

def plot_images(img1, img2, label1, label2, cmap1='viridis', cmap2='viridis'):
    """Plots two images side by side."""
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first image on the first subplot
    axes[0].imshow(img1, cmap=cmap1)
    axes[0].set_title(label1)

    # Plot the second image on the second subplot
    axes[1].imshow(img2, cmap=cmap2)
    axes[1].set_title(label2)

    # Show the plot
    plt.tight_layout()
    plt.show()

def crop_half(img):
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

def connect_lines(img, element_size1=50, element_size2=5):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, element_size1))  # Adjust the kernel size as needed

    # Perform dilation to elongate the white rectangle vertically
    elongated_rectangle = cv2.dilate(img, vertical_kernel, iterations=8)

    # Define the structuring element for the morphology operation (kernel)
    vertical_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (element_size2, 1))  # Adjust the kernel size as needed

    # Perform erosion on the enlongated image
    eroded_img = cv2.morphologyEx(elongated_rectangle, cv2.MORPH_ERODE, vertical_kernel2)
    return eroded_img

def update_trapezoid(bottom_width, top_width, vertical_position, bottom_spacing,
                     horizontal_position, rotation, top_spacing):

    # Convert degrees to radians for rotation
    theta_cw = np.radians(rotation)
    
    # update points based on sliders values
    x1 = horizontal_position - top_width / 2
    x2 = horizontal_position + top_width / 2
    x4 = horizontal_position - bottom_width / 2
    x3 = horizontal_position + bottom_width / 2
    y1 = vertical_position - top_spacing
    y2 = vertical_position - top_spacing
    y4 = vertical_position + bottom_spacing
    y3 = vertical_position + bottom_spacing
    
    # Rotation matrix for clockwise and counterclockwise rotation
    rot = np.array([[np.cos(theta_cw), -np.sin(theta_cw)],
                       [np.sin(theta_cw), np.cos(theta_cw)]])
    
    # Apply rotation transformation to trapezoid vertices
    p1 = np.dot(rot, np.array([[x1], [y1]]))
    p2 = np.dot(rot, np.array([[x2], [y2]]))
    p3 = np.dot(rot, np.array([[x3], [y3]]))
    p4 = np.dot(rot, np.array([[x4], [y4]]))
    pts = np.array([p1, p2, p3, p4], dtype=np.float32)
    return pts