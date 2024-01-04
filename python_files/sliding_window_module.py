from skimage.morphology import dilation
import matplotlib.pyplot as plt
import numpy as np
import cv2

class SlidingWindow:
    """
    Class for lane detection and analysis using sliding windows.

    Attributes:
    window_width (int): Width of the sliding window.
    window_height (int): Height of the sliding window.

    Methods:
    get_lanes_starting_points(img, returnHist=False):
        Calculates starting points of lane lines using histogram analysis.

    sliding_window(img, left_lane_x, right_lane_x, H_img):
        Performs lane detection using sliding windows.

    generate_polynomial_values(coefficients, x_max):
        Generates x and y values for a polynomial defined by its coefficients.

    calculate_curvature(a, b, x):
        Calculates the curvature of a polynomial at a specific point.

    get_direction(left_curve, right_curve, y_value):
        Determines the direction based on lane curvature.

    get_mask(img):
        Creates and processes a mask for lane detection.
    """
    
    def __init__(self, window_width, window_height):
        """
        Initializes the SlidingWindow object with window width and height.

        Args:
        window_width (int): Width of the sliding window.
        window_height (int): Height of the sliding window.
        """
        self.window_width = window_width
        self.window_height = window_height
        self.prev_left_curve_coeff = np.zeros(0)
        self.prev_right_curve_coeff = np.zeros(0)
    
    def get_lanes_starting_points(self, img, returnHist=False):
        """
        Calculates starting points of lane lines using histogram analysis.

        Args:
        img (numpy.ndarray): Input image/frame for lane detection.
        returnHist (bool): Flag to indicate whether to return histogram data.

        Returns:
        tuple or int: Tuple of starting points for left and right lanes (if returnHist=False), 
        or starting points and histogram data (if returnHist=True).
        """
        # Get the Histogram of the bottom half of the frame captured
        half_frame = img.shape[0] //2

        crop_half_frame = img[half_frame:, :]
        axis_Histogram = np.sum(crop_half_frame, axis = 0)
        midpoint = int(axis_Histogram.shape[0] / 2)
        
        # Get where is the max values located on the x-axis
        left_lane_x = np.argmax(axis_Histogram[:midpoint])
        right_lane_x = np.argmax(axis_Histogram[midpoint:]) + midpoint
        if returnHist:
            return left_lane_x, right_lane_x, axis_Histogram
        else:
            return left_lane_x, right_lane_x
    
    
    def sliding_window(self, img,left_lane_x, right_lane_x, H_img):
        """
        Performs lane detection using sliding windows.

        Args:
        img (numpy.ndarray): Input image/frame for lane detection.
        left_lane_x (int): Starting x-coordinate of the left lane line.
        right_lane_x (int): Starting x-coordinate of the right lane line.
        H_img (int): Height of the input image.

        Returns:
        tuple: Coefficients of the fitted polynomials for left and right lanes.
        """
        window_width = self.window_width
        window_height = self.window_height
        
        # Arrays to hold the lane index
        left_lane_list = np.empty((0, 2))
        right_lane_list = np.empty((0, 2))
        
        # Creating the window on the lanes bases
        num_of_windows = H_img // window_height

        # Move the window on the lane
        for i in range(num_of_windows):
            y_start = H_img - window_height * (i + 1)
            y_end = H_img - window_height * i
            
            x_start_left = left_lane_x - window_width//2
            x_end_left = left_lane_x + window_width//2
            
            x_start_right = right_lane_x - window_width//2
            x_end_right = right_lane_x + window_width//2
                    
            left_lane_win = img[y_start : y_end, x_start_left : x_end_left]
            right_lane_win = img[y_start : y_end, x_start_right : x_end_right]

            # get the white pixels in the left and right window
            left_white_pxls = np.argwhere(left_lane_win > 0)
            right_white_pxls = np.argwhere(right_lane_win > 0)

            # move from window coordinates to image coordinates
            left_white_pxls[:, 1] += (left_lane_x - window_width//2)
            left_white_pxls[:, 0] += y_start
            right_white_pxls[:,1] += (right_lane_x - window_width//2)
            right_white_pxls[:, 0] += y_start

            left_lane_list = np.vstack((left_lane_list, left_white_pxls))    
            right_lane_list = np.vstack((right_lane_list, right_white_pxls))    
            # print(left_white_pxls)
            # update the window base
            if(len(left_white_pxls)):
                left_lane_x = int(np.mean(left_white_pxls[:, 1]))
            if(len(right_white_pxls)):
                right_lane_x = int(np.mean(right_white_pxls[:, 1]))

        # Curve fitting(2nd degree poynomial) for the points from the window
        left_curve_coeff, right_curve_coeff = self.prev_left_curve_coeff, self.prev_right_curve_coeff
        if(len(left_lane_list)):        
           left_curve_coeff = np.polyfit(left_lane_list[:, 0], left_lane_list[:, 1], 2)
        if(len(right_lane_list)):
            right_curve_coeff = np.polyfit(right_lane_list[:, 0], right_lane_list[:, 1], 2)

        return left_curve_coeff, right_curve_coeff
        
    def generate_polynomial_values(self, coefficients, x_max):
        """
        Generate x and y values for a polynomial defined by its coefficients.

        Arguments:
        coefficients : array_like
            Coefficients of the polynomial.
        x_max : float
            Maximum value of x for generating values.

        Returns:
        x : ndarray
            Array of x values.
        y : ndarray
            Corresponding array of y values calculated from the polynomial function.
        """
        poly_function = np.poly1d(coefficients)
        
        # generate points from 0 to x max
        x = np.linspace(0, x_max, 100)
        
        # get corresponding y value
        y = poly_function(x)
        
        return x.astype(int), y.astype(int)

    def calculate_curvature(self, a, b, x):
        """
        Calculates the curvature of a polynomial at a specific point.

        Args:
        a (float): Coefficient 'a' of the polynomial.
        b (float): Coefficient 'b' of the polynomial.
        x (float): Point at which the curvature needs to be calculated.

        Returns:
        float: Calculated curvature at the given point.
        """
        numerator = np.abs(2 * a)
        denominator = (1 + (2 * a * x + b)**2) ** (3/2)
        curvature = numerator / denominator
        return curvature

    def get_direction(self):
        """
        Determines the direction based on lane curvature.

        Returns:
        str: Direction ('left', 'right', or 'straight') based on lane curvature.
        """
        # polynomial fitting = Ax^2 + Bx + C
        a_left, b_left, _ = self.prev_left_curve_coeff
        a_right, b_right, _ = self.prev_right_curve_coeff

        # Radius of curvature = ((1 + (2Ax + B)**2)**(3/2)) / |2A|
        L_curvature = self.calculate_curvature(a_left, b_left, 200)
        R_curvature = self.calculate_curvature(a_right, b_right, 200)
        
        # If A = -ve , then the direction of curvature is at the left & vice-versa
        if min(L_curvature, R_curvature) > 1e-5:
            direction = 'right'
        elif min(L_curvature, R_curvature) < -1e-5:
            direction = 'left'
        else:
            direction = 'straight'
        return direction
    
    def get_mask(self, img):
        """
        Creates and processes a mask for lane detection.

        Args:
        img (numpy.ndarray): Input binary image for lane detection.

        Returns:
        numpy.ndarray: Image with the dilated mask overlaid on the original image.
        """
        H, W = img.shape # binary image
        left_x, right_x = self.get_lanes_starting_points(img)
        left_curve_coeff, right_curve_coeff = self.sliding_window(img, left_x, right_x, H)
        
        x_left_list, y_left_list = self.generate_polynomial_values(left_curve_coeff, H - 1)
        x_right_list, y_right_list = self.generate_polynomial_values(right_curve_coeff, H - 1)
        
        # create a mask
        mask = np.zeros((H, W))
        # find indices where 0 < y < width
        indices_left = np.where(np.logical_and(y_left_list > 0, y_left_list < W))[0]
        indices_right = np.where(np.logical_and(y_right_list > 0, y_right_list < W))[0]
        
        # filter x, y
        filtered_x_left = x_left_list[indices_left]
        filtered_y_left = y_left_list[indices_left]
        filtered_x_right = x_right_list[indices_right]
        filtered_y_right = y_right_list[indices_right]
        
        mask[filtered_x_left, filtered_y_left] = 1
        mask[filtered_x_right, filtered_y_right] = 1
        
        # Convert the mask to uint8 format
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Define a kernel for dilation
        kernel = np.ones((30, 30), np.uint8)  # Adjust the kernel size as needed

        # Perform dilation using OpenCV
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

        # Convert the dilated mask to a 3-channel image to match the original image
        dilated_mask = cv2.merge((dilated_mask, dilated_mask*0, dilated_mask*0))
        
        # update previous left and right curve coeff
        self.prev_left_curve_coeff = left_curve_coeff
        self.prev_right_curve_coeff = right_curve_coeff

        return dilated_mask
    
