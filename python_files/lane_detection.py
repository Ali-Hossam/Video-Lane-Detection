import sys
sys.path.append('.')
from python_files.preprocessing_functions import ROI, gray_to_binary, apply_errosion
from python_files.hough_transform_module import Hough
from skimage.morphology import dilation, square
from skimage.color import rgb2gray
import numpy as np
import cv2


class LaneDetection:
    def __init__(self):
        self.hough_ = Hough(5, 500)

        # lines r, theta
        self.past_r1 = 0
        self.past_r2 = 0
        self.past_theta1 = 0
        self.past_theta2 = 0
        
        # distances lists
        self.distances1 = []
        self.distances2 = []

        # current frame
        self.cropped_frame = np.zeros(3)
                
    def calc_distance_between_lines(self, r1, theta1, r2, theta2):
        """calculates distance between two lines given their r and theta in 
            polar coordinates.
        """
        dist = np.abs(r1 - r2 * np.cos(theta1 - theta2))
        return dist

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
        for point in pts_list.astype(int):
            cv2.circle(img_with_points, point, 8, (0, 0, 255), -1)  # Draw red circles (BGR color)
        return img_with_points  # Return the image with scatter points added

    def get_angle(self):
        pass
    
    def detect_lane_frame(self, img, pts_source):
        # crop the image
        cropped_img = ROI(img)
        
        # convert image to gray
        gray_img = rgb2gray(cropped_img)
        
        # apply prespective transformation
        transformed_img, PT_matrix = self.perspective_transformation(gray_img, pts_source) 

        # convert image to binary
        binary_img = gray_to_binary(transformed_img)
        
        # morphological operation
        morph_img = apply_errosion(binary_img, 2)
        
        # apply Hough transform
        r1, theta1, r2, theta2 = self.hough_.get_polar_coorindates(morph_img)
                
        # get full mask
        lane1_mask = self.hough_.get_mask(morph_img, r1, theta1, "green")
        lane2_mask = self.hough_.get_mask(morph_img, r2, theta2, "red")
        lanes_mask = np.maximum(lane1_mask, lane2_mask)
        
        # apply inverse transformation
        H_mask, W_mask, _ = lanes_mask.shape
        inv_lanes_mask = self.inv_perspective_transform(lanes_mask, PT_matrix, W_mask, H_mask)
        
        # apply dilation on the mask
        inv_lanes_mask[:, :, 1] = dilation(inv_lanes_mask[:, :, 1], square(5)) # dilation   

        # add mask to the original frame
        cropped_img[:, :, 1] = np.maximum(cropped_img[:, :, 1], inv_lanes_mask[:, :, 1])
        
        H = img.shape[0]
        img[H//2:H, :] = cropped_img

        # create image with PT points to show it
        img_with_PT_points = self.create_img_with_points(cropped_img, pts_source)
        
        # update current frame
        self.cropped_frame = cropped_img
        return img, morph_img, img_with_PT_points
    
    