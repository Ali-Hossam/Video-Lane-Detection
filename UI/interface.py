import customtkinter as ctk
from ctypes import windll, byref, sizeof, c_int
import cv2
from PIL import Image, ImageTk
import sys
from skimage import transform
sys.path.append('.')
import numpy as np
from python_files.lane_detection import LaneDetection
from python_files.preprocessing_functions import update_trapezoid

ctk.set_appearance_mode("dark")
logo = Image.open("UI/logo.png")
logo = logo.resize((310, 80))
# icon_path = "UI\icon.ico"

YELLOW = "#FFFF33" 
GRAY1 = "#2b2b2b"
GRAY2 = "#242424"
button_height = 50
button_width = 320
slider_width = 300
slider_height = 16
txt_size = 19
header_size = 20
GWL_EXSTYLE = -20
WS_EX_APPWINDOW = 0x00040000
WS_EX_TOOLWINDOW = 0x00000080

# create App class
class App(ctk.CTk):
    # Layout of the GUI will be written in the init
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dimensions of the window
        
        self.geometry("1400x720")
        # self.iconbitmap(icon_path)
        
        #===========================================================#
        # Title bar        
        #===========================================================#
        # remove the title bar
        self.overrideredirect(True)

        self.iconbitmap("UI/logo.png")
        # create a fake titlebar
        title_bar = ctk.CTkFrame(self, height=80, bg_color="Black", corner_radius=0)
        title_bar.pack(side="top", fill="x", anchor="e")
        
        # make fake titlebar move with the mouse
        title_bar.bind("<B1-Motion>", self.move_app)

        # # Add App name
        image_widget = ctk.CTkImage(dark_image=logo, size=(310, 80))
        image_label = ctk.CTkLabel(title_bar, image=image_widget, text="")
        image_label.grid(row=0, column=0, padx=15, pady=10)
        image_label.bind("<B1-Motion>", self.move_app)

        title_bar.grid_columnconfigure(1, weight=1)  # Column 1 (where canvas will be placed) expands

        # Create a canvas widget
        canvas = ctk.CTkCanvas(title_bar, width=80, height=80, bg=GRAY1, highlightthickness=0, confine=True)
        canvas.grid(row=0, column=1, sticky="e")  # Position at the top-right corner

        # Draw an anti-aliased circle on the canvas for the exit button
        circle = canvas.create_aa_circle(40, 40, 20, fill="#F54D4D")  # x, y, radius
        canvas.tag_bind(circle, "<Button-1>", self.quit_app)  # Bind the circle to quit_app function
        #===========================================================#

        #===========================================================#
        # Buttons     
        #===========================================================#
        self.is_hough = True
        self.is_slid = False
        
        alg_label = ctk.CTkLabel(self, text="Algorithms", font=("consolas", header_size, "bold"), 
                             text_color="gray", bg_color=GRAY2)
        alg_label.pack(side="top", anchor="w", padx=20, ipady=20)
        
        self.hough_button_frame1 = ctk.CTkFrame(self, width=button_width, height=button_height, fg_color=YELLOW)
        self.hough_button_frame1.pack(side="top", anchor="nw", padx=0)
        self.hough_button_frame1.pack_propagate(False)
        
        self.hough_button_frame2 = ctk.CTkFrame(self.hough_button_frame1, width=button_width-8, height=button_height, corner_radius=0)
        self.hough_button_frame2.pack(side="top", anchor="nw", padx=0)
        self.hough_button_frame2.pack_propagate(False)
        
        self.hough_button_lbl = ctk.CTkLabel(self.hough_button_frame2, text="Hough Transform", font=("consolas", txt_size, "bold"), 
                             text_color="white")
        self.hough_button_lbl.pack(expand=True)
        self.hough_button_lbl.bind("<Button-1>", self.check_houghB_state)
        self.hough_button_frame2.bind("<Button-1>", self.check_houghB_state)

        # Sliding WIndow button
        self.slid_wind_frame1 = ctk.CTkFrame(self, width=button_width, height=button_height, fg_color=GRAY1)
        self.slid_wind_frame1.pack(side="top", anchor="nw", padx=0, pady=20)
        self.slid_wind_frame1.pack_propagate(False)
        
        self.slid_wind_frame2 = ctk.CTkFrame(self.slid_wind_frame1, width=button_width-8, height=button_height, corner_radius=0)
        self.slid_wind_frame2.pack(side="top", anchor="nw", padx=0)
        self.slid_wind_frame2.pack_propagate(False)

        self.slid_wind_lbl = ctk.CTkLabel(self.slid_wind_frame2, text="Sliding Window", font=("consolas", txt_size, "bold"), 
                             text_color="gray")
        self.slid_wind_lbl.pack(expand=True)
        self.slid_wind_lbl.bind("<Button-1>", self.check_slidingB_state)
        self.slid_wind_frame2.bind("<Button-1>", self.check_slidingB_state)
        
        # upload, start, pause buttons
        bot_button_x = 395
        bot_button_width = button_width // 1.66
        bot_button_space = 25
        bot_button_y = 720 - button_height
        
        self.upload_frame1 = ctk.CTkFrame(self, width=bot_button_width, height=button_height, fg_color=GRAY1)
        self.upload_frame1.place(x=bot_button_x, y=bot_button_y)
        self.upload_frame1.pack_propagate(False)
        
        self.upload_frame2 = ctk.CTkFrame(self.upload_frame1, width=bot_button_width, height=button_height-8, corner_radius=0)
        self.upload_frame2.pack(side="bottom", anchor="center", padx=0)
        self.upload_frame2.pack_propagate(False)
        
        self.start_frame1 = ctk.CTkFrame(self, width=bot_button_width, height=button_height, fg_color=GRAY1)
        self.start_frame1.place(x=bot_button_x + bot_button_width + bot_button_space, y=bot_button_y)
        self.start_frame1.pack_propagate(False)
        
        self.start_frame2 = ctk.CTkFrame(self.start_frame1, width=bot_button_width, height=button_height-8, corner_radius=0)
        self.start_frame2.pack(side="bottom", anchor="center", padx=0)
        self.start_frame2.pack_propagate(False)

        self.pause_frame1 = ctk.CTkFrame(self, width=bot_button_width, height=button_height, fg_color=GRAY1)
        self.pause_frame1.place(x=bot_button_x + 2 * (bot_button_width + bot_button_space), y=bot_button_y)
        self.pause_frame1.pack_propagate(False)
        
        self.pause_frame2 = ctk.CTkFrame(self.pause_frame1, width=bot_button_width, height=button_height-8, corner_radius=0)
        self.pause_frame2.pack(side="bottom", anchor="center", padx=0)
        self.pause_frame2.pack_propagate(False)

        self.upload_label = None
        self.start_label = None
        self.pause_label = None
    
        self.add_label(self.upload_frame2, "Upload", variable="upload_label", is_bind=True, bind_function=self.upload_video)
        self.add_label(self.start_frame2, "Start", variable="start_label", is_bind=True, bind_function=self.start_processing)
        self.add_label(self.pause_frame2, "Pause", variable="pause_label", is_bind=True, bind_function=self.pause_processing)
        
        self.is_start = False
        self.is_pause = False
        self.is_upload = False
        
        self.upload_frame2.bind("<Button-1>", self.upload_video)
        self.start_frame2.bind("<Button-1>", self.start_processing)
        self.pause_frame2.bind("<Button-1>", self.pause_processing)

        #===========================================================#
        
        #===========================================================#
        # Sliders     
        #===========================================================#
        p1_label = ctk.CTkLabel(self, text="Perspective transformation", font=("consolas", header_size, "bold"), 
                             text_color="gray", bg_color=GRAY2)
        p1_label.pack(side="top", anchor="nw", padx=20, pady=20)
        
        self.prespT_param_frame = ctk.CTkFrame(self, width=button_width-8, height=180, corner_radius=5)
        self.prespT_param_frame.pack(side="top", anchor="nw", padx=0)  
        
        self.topWidth_S = self.add_slider(0, "Top Width", 1, 640, 640//8)
        self.bottomWidth_S = self.add_slider(2, "Bottom Width", 1, 640, 640//5)
        self.top_spacing_S = self.add_slider(4, "Top line spacing", 1, 400, 640//5)
        self.bottom_spacing_S = self.add_slider(6, "Bottom line spacing", 1, 400, 0)
        self.horizontal_S = self.add_slider(8, "Horizontal Position", 0, 640, 640//2)
        self.vertical_S = self.add_slider(10, "Vertical Position", 0, 640, 400//2)
        self.rotation_S = self.add_slider(12, "Rotation", -180, 180, 0)
        
        binary_threshold_frame = ctk.CTkFrame(self, width = 400, height=60)
        binary_threshold_frame.place(x=1060, y=720-100)
        binary_th_label = ctk.CTkLabel(binary_threshold_frame, text="Binary Threshold", 
                                       font=("consolas", header_size, "bold"), 
                             text_color="white", bg_color=GRAY2)
        binary_th_label.grid(row=0, pady=3, sticky="w", padx=20)
        self.binary_thresh = ctk.CTkSlider(binary_threshold_frame, from_=0, to=255, progress_color=YELLOW, 
                                    button_color="black", height=slider_height,
                                    width=slider_width - 20, button_hover_color="gray",
                                    command=self.update_parameters)
                
        self.binary_thresh.grid(row=1, padx=10, pady=3)        
        binary_threshold_frame.grid_propagate(False)
        #===========================================================#
        
        #===========================================================#
        # Video play     
        #===========================================================#
        # Define frames
        main_vid_frame = ctk.CTkFrame(self, width=640, height=400, corner_radius=5)
        PT_vid_frame = ctk.CTkFrame(self, width=300, height=195, corner_radius=5)
        lane_vid_frame = ctk.CTkFrame(self, width = 300, height=195, corner_radius=5)
        
        # Define frames position
        main_vid_frame.place(x=360, y=165)
        PT_vid_frame.place(x=1060, y=165)
        lane_vid_frame.place(x=1060, y=390)
        
        # Add label to each frame
        self.main_vid_label = ctk.CTkLabel(main_vid_frame, width=640, height=400, text="") # the label that will hold the video
        self.PT_vid_label = ctk.CTkLabel(PT_vid_frame, width=300, height=210, text="") # the label that will hold the video
        self.lane_vid_label = ctk.CTkLabel(lane_vid_frame, width=300, height=210, text="") # the label that will hold the video
        
        # Define padding for each label
        self.main_vid_label.pack(padx=10, pady=10)
        self.PT_vid_label.pack(padx=10, pady=10)
        self.lane_vid_label.pack(padx=10, pady=10)
        
        # Disable pack propagation
        PT_vid_frame.pack_propagate(False)
        lane_vid_frame.pack_propagate(False)
        
        # Define resizing data
        self.vid_size = (640, 400)

        #===========================================================#


    def add_image_to_frame(self, img, frame_label, size):
        """
        Adds an image (one frame from the video) to a label UI frame.

        Arguments:
        img : numpy.ndarray
            Image data in the form of a NumPy array.
        frame_label : str
            Name of the label UI frame to which the image will be added.
        size : tuple
            Size of the image.

        Notes:
        - Converts BGR image to RGB if the image has more than 2 dimensions.
        - Utilizes CTkImage and configures the specified label frame with the image.
        """
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ctk.CTkImage(img, size=size)
        vid_label_func = getattr(self, frame_label)
        vid_label_func.configure(image=img)
        setattr(self, frame_label, vid_label_func)

    def upload_video(self, e):
        """
        Event handler for uploading a video.

        Arguments:
        e : Event
            Event object.

        Notes:
        - Updates the appearance of the upload button.
        - Opens a file dialog to select the video file.
        - Reads the video file and stores the first frame resized to (640, 400) pixels.
        - Adds the first frame to the main video label frame.
        - Calls other methods to define lane detection models and update parameters.
        """
        # Update button appearance
        self.upload_label.configure(text_color="white")
        self.upload_frame1.configure(fg_color=YELLOW)
        
        # Store file path and open video capture
        vid_path = ctk.filedialog.askopenfilename()
        self.video = cv2.VideoCapture(vid_path)
        ret, frame = self.video.read()
        self.first_frame = cv2.resize(frame, (640, 400))
        self.add_image_to_frame(self.first_frame, "main_vid_label", self.vid_size)
        self.define_lane_detection_model()
        self.update_parameters()
        
        # Update button appearance
        self.upload_label.configure(text_color="gray")
        self.upload_frame1.configure(fg_color=GRAY1)

        
    def define_lane_detection_model(self):
        """
        Defines the lane detection model based on selected algorithms.

        Notes:
        - Checks the flags is_hough and is_slid to select the lane detection algorithm.
        - Initializes the model based on the selected algorithm (Hough or Sliding Window).
        """
        if self.is_hough:
            self.model = LaneDetection("hough")
        if self.is_slid:
            self.model = LaneDetection("slidingW")

    def start_processing(self, e):
        """
        Event handler for starting/stopping video processing.

        Arguments:
        e : Event
            Event object.

        Notes:
        - If is_start is True, stops processing and updates button appearance to 'stop'.
        - If is_start is False, starts processing, updates button appearance to 'start',
        and calls the process_next_frame method.
        """
        if self.is_start == True:
            # Stop processing
            self.start_label.configure(text_color="gray")
            self.start_frame1.configure(fg_color=GRAY1)
            self.is_start = False
        else:
            # Start processing
            self.start_label.configure(text_color="white")
            self.start_frame1.configure(fg_color=YELLOW)
            self.process_next_frame()  # Assuming this method starts processing the next frame
            self.is_start = True


    def update_parameters(self, e=None):
        """
        Updates the parameters and visual representations based on user input.

        Arguments:
        e : Event, optional
            Event object.

        Returns:
        ndarray
            Array of updated points for the trapezoid.

        Notes:
        - Retrieves parameter values from the corresponding UI sliders.
        - Calculates points_array using the update_trapezoid function with the obtained parameters.
        - Updates images based on the processed frames and parameters.
        - Displays the perspective-transformed and binary thresholded images.
        - Returns the updated points_array for the trapezoid.
        """
        top_width = int(self.topWidth_S.get())
        bottom_width = int(self.bottomWidth_S.get())
        top_spacing = int(self.top_spacing_S.get())
        bottom_spacing = int(self.bottom_spacing_S.get())
        horizontal_pos = int(self.horizontal_S.get())
        vertical_pos = int(self.vertical_S.get())
        rotation = int(self.rotation_S.get())
            
        points_array = update_trapezoid(bottom_width, top_width, vertical_pos,
                                bottom_spacing, horizontal_pos, rotation, top_spacing)
        
        if self.model.current_frame.any():
            img = self.model.current_frame
        else:
            img = cv2.resize(self.first_frame, (640, 400)) 
        img_pts = self.model.create_img_with_points(img, points_array)
        self.add_image_to_frame(img_pts, "PT_vid_label", (280, 175))
        
        # Show perspective-transformed image
        threshold = int(self.binary_thresh.get())
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pt_img, _ = self.model.perspective_transformation(gray_img, points_array)
        _, binary_img = cv2.threshold(pt_img, threshold, 255, cv2.THRESH_BINARY)
        self.add_image_to_frame(binary_img, "lane_vid_label", (280, 175))
        
        return points_array

     
    def process_next_frame(self):
        """
        Processes the next frame of the video for lane detection and updates UI.

        Notes:
        - If not paused, reads the next frame from the video.
        - Updates parameters using update_parameters.
        - Detects lanes in the frame and displays processed images.
        - Continuously calls itself using 'after' method for continuous processing.
        """
        if not self.is_pause:
            ret, frame = self.video.read()
            points_array = self.update_parameters()
            
            # Show video frames after processing
            if not ret:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Set frame position to beginning
                return
            frame = cv2.resize(frame, (640, 400))
            img, morph_img, img_pts = self.model.detect_lane_frame(frame, points_array, int(self.binary_thresh.get()))
            self.add_image_to_frame(img, "main_vid_label", self.vid_size)        
            self.add_image_to_frame(img_pts, "PT_vid_label", (280, 175))
            self.add_image_to_frame(morph_img, "lane_vid_label", (280, 175))
            self.update()
            self.after(1, self.process_next_frame())  # Continuously process next frame

    def pause_processing(self, e):
        """
        Pauses or resumes video processing based on the current state.

        Arguments:
        e : Event
            Event object.

        Notes:
        - If is_pause is True, resumes processing and updates button appearance.
        - If is_pause is False, pauses processing and updates button appearance.
        """
        if self.is_pause == True:
            self.pause_label.configure(text_color="gray")
            self.pause_frame1.configure(fg_color=GRAY1)
            self.is_pause = False
            self.process_next_frame()
        else:
            self.pause_label.configure(text_color="white")
            self.pause_frame1.configure(fg_color=YELLOW)
            self.is_pause = True

    def add_label(self, frame, text, variable, font_size=txt_size, is_bind=False, bind_function=None):
        """
        Adds a label to a given frame in the UI.

        Arguments:
        frame : Frame
            Frame where the label will be added.
        text : str
            Text content of the label.
        variable : str
            Name of the label variable.
        font_size : int, optional
            Font size of the label text. Default is txt_size.
        is_bind : bool, optional
            Boolean indicating if the label has a binding function. Default is False.
        bind_function : function, optional
            Function to be executed on label click. Default is None.
        """
        label = ctk.CTkLabel(frame, text=text, font=("consolas", font_size, "bold"), text_color="gray")
        label.pack(expand=True)
        if is_bind:
            label.bind("<Button-1>", bind_function)
        setattr(self, variable, label)

    def add_slider(self, row, text, start, end, value, col=0):
        """
        Adds a slider to the UI with specified parameters.

        Arguments:
        row : int
            Row position in the UI grid.
        text : str
            Text label for the slider.
        start : int
            Start value for the slider.
        end : int
            End value for the slider.
        value : int
            Initial value for the slider.
        col : int, optional
            Column position in the UI grid. Default is 0.

        Returns:
        ctk.CTkSlider
            Slider object created in the UI.

        Notes:
        - Adds a slider with specified text, range, and initial value to the UI grid.
        - Binds the slider to the update_parameters method for value changes.
        """
        slider1_label = ctk.CTkLabel(self.prespT_param_frame, text=text, font=("consolas", txt_size, "bold"), 
                            text_color="white", bg_color=GRAY1)
        slider1_label.grid(row=row, column=col, padx=20, pady=1, sticky='w')
        
        slider = ctk.CTkSlider(self.prespT_param_frame, from_=start, to=end, progress_color=YELLOW, 
                                    button_color="black", height=slider_height,
                                    width=slider_width, button_hover_color="gray",
                                    command=self.update_parameters)
        slider.grid(row=row+1, column=col, padx=10)
        slider.set(value)
        return slider

        
    def move_app(self, e):
        """moves the app with clicks on the titlebar."""
        self.geometry(f'+{e.x_root}+{e.y_root}')
    
    def quit_app(self, e):
        """Quits the app!"""
        self.quit()
        self.destroy()   
    
    def check_houghB_state(self, e):
        """Changes the state of hough transform button if pressed."""
        if self.is_slid:
            self.slid_wind_frame1.configure(fg_color=GRAY1)
            self.slid_wind_lbl.configure(text_color="gray")
            self.is_slid = False
            
        self.hough_button_frame1.configure(fg_color=YELLOW)
        self.hough_button_lbl.configure(text_color="white")
        self.is_hough = True
        self.define_lane_detection_model()
            
    def check_slidingB_state(self, e):
        """changes the state of Sliding window button if pressed."""
        if self.is_hough:
            self.hough_button_frame1.configure(fg_color=GRAY1)
            self.hough_button_lbl.configure(text_color="gray")
            self.is_hough = False
            
        self.slid_wind_frame1.configure(fg_color=YELLOW)
        self.slid_wind_lbl.configure(text_color="white")
        self.is_slid = True
        self.define_lane_detection_model()
    

def show_taskbar_icon(root):
    hwnd = windll.user32.GetParent(root.winfo_id())
    style = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style = style & ~WS_EX_TOOLWINDOW
    style = style | WS_EX_APPWINDOW
    res = windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    # re-assert the new window style
    root.wm_withdraw()
    root.after(10, lambda: root.wm_deiconify())
       
if __name__ == "__main__":
    
    app = App()
    # Runs the loop
    app.after(10, lambda: show_taskbar_icon(app))
    app.mainloop()
    
