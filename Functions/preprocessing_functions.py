import numpy as np

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

