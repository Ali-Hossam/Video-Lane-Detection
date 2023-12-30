## Functions here can differ a little from the Notebook!

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from skimage.morphology import dilation, square

class Hough:
    def __init__(self, thetas_step, kernel_size):
        """
        Args:
            - thetas_step (int): Step size for theta values.
            - kernel_size (int): Size of the kernel for neighbors removal.
        """
        self.thetas_step = thetas_step
        self.kernel_size = kernel_size
    
    
    def get_non_zero_pixels(self, img: np.ndarray) -> np.ndarray:
        """
        Returns the indices of non-zero pixels in a grayscale image.

        Parameters:
            img (np.ndarray): A 2D array representing a grayscale image.

        Returns:
            np.ndarray: An array containing row and column indices of non-zero pixels.
        """
        return np.stack(np.where(img > 0), axis=1)
    
    def calc_r_vec(self, thetas: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates r values in polar coordinates for multiple x, y, and thetas using the equation:
        r = x * np.cos(theta) + y * np.sin(theta).
        
        Parameters:
            thetas (np.ndarray): Array of thetas in degrees.
            x (np.ndarray): Array of x values.
            y (np.ndarray): Array of y values.
        
        Returns:
            np.ndarray: Array of polar coordinates r values.
        """
        theta_rad = np.deg2rad(thetas)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        r = np.outer(x, cos_theta) + np.outer(y, sin_theta)
        return r.astype(int)
    
    def get_max_value_idx(self, data: np.ndarray) -> np.ndarray:
        """
        Finds the indices of the maximum value in a 2D array.

        Parameters:
        - data (np.ndarray): Input 2D array.

        Returns:
        - np.ndarray: Indices of the maximum value in the input array.
        """
        sorted_arr = np.sort(data.reshape(-1))
        return np.argwhere(data == sorted_arr[-1])[0]
    
    def remove_neighbors(self, data: np.ndarray, theta: int, r: int, kernel_size: int) -> np.ndarray:
        """
        Nullifies value at (theta, r) and its neighbors within a specified kernel size.
        
        Parameters:
            data (np.ndarray): A 2D array where thetas are rows and columns are r.
            theta (int): Index along the theta dimension.
            r (int): Index along the r dimension.
            kernel_size (int): Size of the kernel.
        
        Returns:
            np.ndarray: Filtered array.
        """
        half_kernel = kernel_size // 2

        start_th = np.clip(theta - half_kernel, 0, data.shape[0])
        end_th = np.clip(theta + half_kernel + 1, 0, data.shape[0])

        start_r = np.clip(r - half_kernel, 0, data.shape[1])
        end_r = np.clip(r + half_kernel + 1, 0, data.shape[1])

        data[start_th:end_th, start_r:end_r] = 0
        return data
    
    def get_polar_coorindates(self, img):
        """
        Performs Hough Transform on an image to detect peaks and return their polar coordinates.

        Parameters:
        - img (np.ndarray): Input image.
        
        Returns:
            Polar coordinates of the two lanes
            r1, theta1, r2, theta2.
        """
        thetas_step = self.thetas_step
        kernel_size = self.kernel_size
        
        # Calculate image dimensions
        h, w = img.shape
        D = int(np.sqrt(h**2 + w**2))
        
        # Create Hough array
        h_arr = np.zeros((180, 2*D))

        # Get coordinates of non-zero pixels
        non_zero_pxls = self.get_non_zero_pixels(img)
        
        # Calculate r
        thetas = range(0, 180, thetas_step)
        x = non_zero_pxls[:, 0]
        y = non_zero_pxls[:, 1]
        r = self.calc_r_vec(thetas, x, y) + D
        np.add.at(h_arr, (thetas, r), 1)  # Update h_arr
        
        new_h_arr = h_arr.copy()
        
        # Get first peak
        theta1, r1 = self.get_max_value_idx(new_h_arr)
        
        # Remove max peak neighbors
        new_h_arr = self.remove_neighbors(new_h_arr, theta1, r1, kernel_size)
        
        # Get second peak
        theta2, r2 = self.get_max_value_idx(new_h_arr)
        r1 -= D
        r2 -= D
        
        # Return polar coordinates of the two lanes lines
        return r1, theta1, r2, theta2

    def polar_to_cartesian_y(self, x: np.ndarray, r: int, theta: int) -> np.ndarray:
        """
        Calculates Cartesian y values given x and (r, theta) of the polar coordinates.
        
        Parameters:
            x (np.ndarray): Array of x values.
            r (int): Value of r.
            theta (int): Value of theta in degrees.
            
        Returns:
            np.ndarray: Calculated Cartesian y values.
        """
        theta_rad = np.deg2rad(theta)
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)
        
        y = (r - x * cos_theta) / (sin_theta + 1e-9)
        return y.astype(int)

    def draw_line(self, img: np.ndarray, r: int, theta: int, color: str) -> np.ndarray:
        """
        Draws a line on an image given polar coordinates.

        Parameters:
        - img (np.ndarray): Input image.
        - r (int): Polar coordinate r.
        - theta (int): Polar coordinate theta.

        Returns:
        - np.ndarray: Image with drawn line.
        """
        if(len(img.shape) == 2):
            img = gray2rgb(img)
            
        height, width, c = img.shape
        mask = np.zeros(img.shape)
        x = np.array(range(height))
        y = self.polar_to_cartesian_y(x, r, theta)
        
        # find indices where 0 < y < width
        indices = np.where(np.logical_and(y > 0, y < width))[0]
        
        # filter x, y
        filtered_x = x[indices]
        filtered_y = y[indices]
        
        if color == "green":
            mask[filtered_x, filtered_y] = [0, 254, 0]  # Set pixel color (green) on the line
        elif color == "red":
            mask[filtered_x, filtered_y] = [0, 220, 0]  # Set pixel color (green) on the line
        mask[:, :, 1] = dilation(mask[:, :, 1], square(5))
        # return np.maximum(img, mask)
        return mask
        
    def get_mask(self, img, r, theta, color):
        """
        Applies the Hough transform process on the input image to detect road lines.
        It generates lines using Hough transform.

        Parameters:
        - img (np.ndarray): Input image.

        Returns:
            Polar coordinates of the lanes (r1, theta1, r2, theta2)
        """
        line_image = np.zeros((img.shape[0], img.shape[1], 3))
        line_image = self.draw_line(gray2rgb(img), r, theta, color)
        # line_image = np.maximum(line_image, self.draw_line(line_image, r2, theta2))
        # line_image = line_image / np.max(line_image) 
        
        # # Visualization
        # plt.figure(figsize=(12, 5))
        # plt.imshow(line_image) # clipping may occur
        # plt.tight_layout()
        # plt.show()
        return line_image
