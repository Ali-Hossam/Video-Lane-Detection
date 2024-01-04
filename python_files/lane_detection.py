import sys
sys.path.append('.')
from python_files.preprocessing_functions import gray_to_binary, connect_lines
from python_files.hough_transform_module import Hough
from python_files.sliding_window_module import SlidingWindow
from skimage.morphology import dilation, square
from skimage.color import rgb2gray
import numpy as np
import cv2


class LaneDetection:
    def __init__(self, algorithm):
        if algorithm == "hough":
            self.algorithm = 0
            self.model = Hough(5, 500)
        else:
            self.algorithm = 1
            self.model = SlidingWindow(50, 30)

        # current frame
        self.current_frame = np.zeros(3)


    def perspective_transformation(self, img, pts_source):
        """Returns perspective transformation of an image."""
        H, W = img.shape
        
        # Define source points, points are defined as (col, row)
        d_top_left = (0, 0)
        d_top_right = (W-1, 0)
        d_bottom_left = (0, H - 1)
        d_bottom_right = (W - 1, H - 1)
        
        pts_destination = np.array([d_top_left, d_top_right, d_bottom_right, d_bottom_left], dtype=np.float32)
          
        # Get perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(pts_source, pts_destination)
        
        # Apply perspective transformation
        result = cv2.warpPerspective(img, matrix, (W, H))
        return result, matrix

    def inv_perspective_transform(self, inv_frame, t_matrix, W, H):
        """Applies Inverse Perspective Transformation on a transformed frame."""
        inv_t_matrix= np.linalg.inv(t_matrix)
        frame = cv2.warpPerspective(inv_frame, inv_t_matrix, (W, H))
        return frame

    def create_img_with_points(self, img, pts_list):
        """draw Prespective transformation points on an image"""
        img_with_points = np.copy(img)
        # Iterate through points to draw circles
        pts_list = pts_list.reshape(4, 2).astype(int)
        
        for point in pts_list:
            cv2.circle(img_with_points, point, 8, (200, 0, 0), -1)  # Draw red circles (BGR color)

        # Draw lines between consecutive points
        for i in range(len(pts_list) - 1):
            cv2.line(img_with_points, pts_list[i], pts_list[i + 1], (150, 0, 0), 2)  # Draw green lines

        # To close the shape, draw a line between the last and first points
        cv2.line(img_with_points, pts_list[-1], pts_list[0], (150, 0, 0), 2)
    
        # Return the image with scatter points added
        return img_with_points
    
    def get_angle(self):
        pass
    
    def detect_lane_frame(self, img, pts_source, binary_threshold):
        # resize the image
        resized_img = cv2.resize(img, (640, 400))
        
        # convert image to gray
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
        # apply prespective transformation
        transformed_img, PT_matrix = self.perspective_transformation(gray_img, pts_source) 

        # convert image to binary
        _, binary_img = cv2.threshold(transformed_img, binary_threshold, 255, cv2.THRESH_BINARY)
        
        # morphological operation
        morph_img = connect_lines(binary_img)
        
        if(self. algorithm == 0):
            # apply Hough transform
            r1, theta1, r2, theta2 = self.model.get_polar_coorindates(morph_img)
                
            # get full mask
            lanes_mask = self.model.get_mask(morph_img, r1, theta1, r2, theta2)
        else:
            lanes_mask = self.model.get_mask(morph_img)
            
        # apply inverse transformation
        H_mask, W_mask, _ = lanes_mask.shape
        inv_lanes_mask = self.inv_perspective_transform(lanes_mask, PT_matrix, W_mask, H_mask)
        
        # add mask to the original frame
        resized_img = cv2.addWeighted(resized_img, 0.8, inv_lanes_mask, 1, 0)

        # create image with PT points to show it
        img_with_PT_points = self.create_img_with_points(resized_img, pts_source)
        
        # update current frame
        self.current_frame = resized_img
        return resized_img, binary_img, img_with_PT_points
    
    