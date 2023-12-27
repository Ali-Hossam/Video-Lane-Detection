import customtkinter as ctk
from ctypes import windll, byref, sizeof, c_int
import pywinstyles
import cv2
from PIL import Image, ImageTk
import sys
from skimage import transform
sys.path.append('.')
import numpy as np
from python_files.lane_detection import LaneDetection
from python_files.preprocessing_functions import ROI

ctk.set_appearance_mode("dark")
logo = Image.open("UI/logo.png")
logo = logo.resize((310, 80))
icon_path = "UI\icon.ico"

YELLOW = "#FFFF33" 
GRAY1 = "#2b2b2b"
GRAY2 = "#242424"
button_height = 50
button_width = 320
slider_width = 100
slider_height = 16
txt_size = 20
header_size = 22
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
        self.iconbitmap(icon_path)
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
        
        param_label = ctk.CTkLabel(self, text="Parameters", font=("consolas", header_size, "bold"), 
                             text_color="gray", bg_color=GRAY2)
        param_label.pack(side="top", anchor="w", padx=20, ipady=20)
        
        p1_label = ctk.CTkLabel(self, text="Prespective tranformation", font=("consolas", txt_size, "bold"), 
                             text_color="white", bg_color=GRAY2)
        p1_label.pack(side="top", anchor="nw", padx=20)
        
        self.prespT_param_frame = ctk.CTkFrame(self, width=button_width-8, height=180, corner_radius=0)
        self.prespT_param_frame.pack(side="top", anchor="nw", padx=0)  
        
        self.sliderx1, self.slidery1 = self.add_two_slidersH(1)
        self.sliderx2, self.slidery2 = self.add_two_slidersH(2)
        self.sliderx3, self.slidery3 = self.add_two_slidersH(3)
        self.sliderx4, self.slidery4 = self.add_two_slidersH(4)
        
        self.sliderx1.configure(to=639)
        self.sliderx2.configure(to=639)
        self.sliderx3.configure(to=639)
        self.sliderx4.configure(to=639)
        self.slidery1.configure(to=199)
        self.slidery2.configure(to=199)
        self.slidery3.configure(to=199)
        self.slidery4.configure(to=199)
        
        param_data_frame = ctk.CTkFrame(self, height=100, width=button_width+5)
        param_data_frame.pack(side="left", anchor="sw")
        txt = "Point1 : Top left    \nPoint2 : Top right   \nPoint3 : bottom right\nPoint4 : bottom left "
        param_data_label = ctk.CTkLabel(param_data_frame, text=txt, 
                                        font=("consolas", txt_size, "bold"),
                                        text_color="gray")
        param_data_label.pack(anchor="center")
        param_data_frame.pack_propagate(False)
        #===========================================================#
        
        #===========================================================#
        # Video play     
        #===========================================================#
        # frames
        main_vid_frame = ctk.CTkFrame(self, width=640, height=400, corner_radius=20)
        main_vid_frame.place(x=360, y=165)
        self.main_vid_label = ctk.CTkLabel(main_vid_frame, width=640, height=400, text="") # the label that will hold the video
        self.main_vid_label.pack(padx=20, pady=20)
        
        PT_vid_frame = ctk.CTkFrame(self, width=300, height=210, corner_radius=20)
        PT_vid_frame.place(x=1060, y=165)
        self.PT_vid_label = ctk.CTkLabel(PT_vid_frame, width=300, height=210, text="") # the label that will hold the video
        self.PT_vid_label.pack(padx=20, pady=20)
        lane_vid_frame = ctk.CTkFrame(self, width = 300, height=210, corner_radius=20)
        lane_vid_frame.place(x=1060, y=395)
        self.lane_vid_label = ctk.CTkLabel(lane_vid_frame, width=300, height=210, text="") # the label that will hold the video
        self.lane_vid_label.pack(padx=20, pady=20)
        PT_vid_frame.pack_propagate(False)
        lane_vid_frame.pack_propagate(False)
        
        self.vid_size = (640, 400)

        cap = cv2.VideoCapture("images\LaneVideo.mp4")
        self.video = cap
        
        # self.video_play()
        #===========================================================#

    
    def add_image_to_frame(self, img, frame_label, size):
        """Adds an image(one frame from the video) to a label in a frame of the UI."""
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ctk.CTkImage(img, size=size)
        vid_label_func = getattr(self, frame_label)
        vid_label_func.configure(image=img)
        setattr(self, frame_label, vid_label_func)
        
    def upload_video(self, e):
        # update button appearance
        self.upload_label.configure(text_color="white")
        self.upload_frame1.configure(fg_color=YELLOW)
        
        # store file path
        vid_path = ctk.filedialog.askopenfilename()
        self.video = cv2.VideoCapture(vid_path)
        ret, frame = self.video.read()
        self.first_frame = cv2.resize(frame, (640, 400))
        self.add_image_to_frame(self.first_frame, "main_vid_label", self.vid_size)
        self.define_lane_detection_model()

        # update button appearance
        self.upload_label.configure(text_color="gray")
        self.upload_frame1.configure(fg_color=GRAY1)
        
    def define_lane_detection_model(self):
        if self.is_hough:
            self.model = LaneDetection()    
        
    def start_processing(self, e):
        if self.is_start == True:
            self.start_label.configure(text_color="gray")
            self.start_frame1.configure(fg_color=GRAY1)
            self.is_start = False
        else:
            self.start_label.configure(text_color="white")
            self.start_frame1.configure(fg_color=YELLOW)
            self.process_next_frame()
            self.is_start = True

    def update_parameters(self, _=None):
        x_values = (
                int(self.sliderx1.get()),
                int(self.sliderx2.get()),
                int(self.sliderx3.get()),
                int(self.sliderx4.get())
                )

        y_values = (
                int(self.slidery1.get()),
                int(self.slidery2.get()),
                int(self.slidery3.get()),
                int(self.slidery4.get())
            )
            
        points_array = [(x, y) for x, y in zip(x_values, y_values)]
        points_array = np.array(points_array, dtype=np.float32)
        # if(self.model.cropped_frame):
        #     img_pts = self.model.cropped_frame
        # else:
        
        if (self.model.cropped_frame.any()):
            img_pts = self.model.cropped_frame
        else:
            img_pts = ROI(self.first_frame) 
        img_pts = self.model.create_img_with_points(img_pts, points_array)
        self.add_image_to_frame(img_pts, "PT_vid_label", (260, 170))

        
        return points_array
    
    def process_next_frame(self):
        if self.is_hough:
            if not self.is_pause:
                ret, frame = self.video.read()
                points_array = self.update_parameters()
                
                # show video frames after processing
                if not ret:
                    self.video.release()
                    cv2.destroyAllWindows()
                    return
                frame = cv2.resize(frame, (640, 400))
                img, morph_img, img_pts = self.model.detect_lane_frame(frame, points_array)
                self.add_image_to_frame(img, "main_vid_label", self.vid_size)        
                self.add_image_to_frame(img_pts, "PT_vid_label", (260, 170))
                self.add_image_to_frame(morph_img, "lane_vid_label", (260, 170))
                self.update()  #
                self.after(1, self.process_next_frame())
            
            
        
    def pause_processing(self, e):
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
        """Adds a label to a given frame."""
        label = ctk.CTkLabel(frame, text=text, font=("consolas", font_size, "bold"), 
                             text_color="gray")
        label.pack(expand=True)
        if is_bind:
            label.bind("<Button-1>", bind_function)
        setattr(self, variable, label)
    
    def add_two_slidersH(self, row):  # add a variable to take
        slider1_label = ctk.CTkLabel(self.prespT_param_frame, text=f"x{row}", font=("consolas", txt_size, "bold"), 
                            text_color="white", bg_color=GRAY1)
        slider1_label.grid(row=row, column=0, padx = 20, pady=10)
        
        slider1 = ctk.CTkSlider(self.prespT_param_frame, from_=0, to=100, progress_color=YELLOW, 
                                    button_color="black", height=slider_height,
                                    width=slider_width, button_hover_color="gray",
                                    command=self.update_parameters)
        slider1.grid(row=row, column=1)
        
        slider2_label = ctk.CTkLabel(self.prespT_param_frame, text=f"y{row}", font=("consolas", txt_size, "bold"), 
                            text_color="white", bg_color=GRAY1)
        slider2_label.grid(row=row, column=2, padx = 20, pady=10)
        
        slider2 = ctk.CTkSlider(self.prespT_param_frame, from_=0, to=100, progress_color=YELLOW, 
                                    button_color="black", height=slider_height,
                                    width=slider_width, button_hover_color="gray",
                                    command=self.update_parameters)
        slider2.grid(row=row, column=3)
        return slider1, slider2
        
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
            
    def check_slidingB_state(self, e):
        """changes the state of Sliding window button if pressed."""
        if self.is_hough:
            self.hough_button_frame1.configure(fg_color=GRAY1)
            self.hough_button_lbl.configure(text_color="gray")
            self.is_hough = False
            
        self.slid_wind_frame1.configure(fg_color=YELLOW)
        self.slid_wind_lbl.configure(text_color="white")
        self.is_slid = True
    

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
    
