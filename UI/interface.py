import tkinter as tk
import customtkinter as ctk
from tkVideoPlayer import TkinterVideo
from ctypes import windll, byref, sizeof, c_int
import pywinstyles
from PIL import Image
import cv2

ctk.set_appearance_mode("dark")
logo = Image.open("UI/logo.png")
logo = logo.resize((310, 80))
YELLOW = "#FFF000"
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
        #===========================================================#
        
        #===========================================================#
        # Sliders     
        #===========================================================#
        self.param1 = 0.0
        self.param2 = 0.0
        param_label = ctk.CTkLabel(self, text="Parameters", font=("consolas", header_size, "bold"), 
                             text_color="gray", bg_color=GRAY2)
        param_label.pack(side="top", anchor="w", padx=20, ipady=20)
        
        p1_label = ctk.CTkLabel(self, text="Prespective tranformation", font=("consolas", txt_size, "bold"), 
                             text_color="white", bg_color=GRAY2)
        p1_label.pack(side="top", anchor="nw", padx=20)
        
        self.prespT_param_frame = ctk.CTkFrame(self, width=button_width-8, height=180, corner_radius=0)
        self.prespT_param_frame.pack(side="top", anchor="nw", padx=0)  
        
        self.add_two_slidersH(1)
        self.add_two_slidersH(2)
        self.add_two_slidersH(3)
        self.add_two_slidersH(4)
        #===========================================================#
        
        #===========================================================#
        # Video play     
        #===========================================================#
        # frames
        main_vid_frame = ctk.CTkFrame(self, width=640, height=400, corner_radius=20)
        main_vid_frame.place(x=380, y=165)

        # PT_vid_frame = ctk.CTkFrame(self, width=300, height=188, corner_radius=20)
        # PT_vid_frame.place(x=1050, y=165)

        # lane_vid_frame = ctk.CTkFrame(self, width = 300, height=188, corner_radius=20)
        # lane_vid_frame.place(x=1050, y=375)
        cap = cv2.VideoCapture("images\LaneVideo.mp4")
        self.frame = main_vid_frame
        self.video = cap
        self.vid_size = (640, 400)
        # self.video_play()
        
    # def video_play(self):
    #     self.frame
    #     self.video
    #     self.vid_size
    #     self.video_label = ctk.CTkLabel(self.frame, text="")
    #     self.video_label.pack(padx=20, pady=20)
    #     ret, frame = self.video.read()
    #     if not ret:
    #         return
    #     resized_frame = cv2.resize(self.frame, self.size)
    #     resized_frame = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    #     # Convert the frame to PhotoImage
    #     img = ctk.CTkImage(dark_image=resized_frame,size=self.size)
    #     self.video_label.configure(image=img)
    #     self.video_label.image = img  # Keep a reference to avoid garbage collection issues
    #     self.update()  #
    #     self.after(1, self.video_play()) 

    #     self.video.release()   
        
             
    def add_two_slidersH(self, row):  # add a variable to take
        slider1_label = ctk.CTkLabel(self.prespT_param_frame, text=f"x{row}", font=("consolas", txt_size, "bold"), 
                            text_color="white", bg_color=GRAY1)
        slider1_label.grid(row=row, column=0, padx = 20, pady=10)
        
        self.slider1 = ctk.CTkSlider(self.prespT_param_frame, from_=0, to=100, progress_color=YELLOW, 
                                    button_color="black", height=slider_height,
                                    width=slider_width, button_hover_color="gray")
        self.slider1.grid(row=row, column=1)
        
        slider2_label = ctk.CTkLabel(self.prespT_param_frame, text=f"y{row}", font=("consolas", txt_size, "bold"), 
                            text_color="white", bg_color=GRAY1)
        slider2_label.grid(row=row, column=2, padx = 20, pady=10)
        
        self.slider2 = ctk.CTkSlider(self.prespT_param_frame, from_=0, to=100, progress_color=YELLOW, 
                                    button_color="black", height=slider_height,
                                    width=slider_width, button_hover_color="gray")
        self.slider2.grid(row=row, column=3)
        
    def move_app(self, e):
        """moves the app with clicks on the titlebar."""
        self.geometry(f'+{e.x_root}+{e.y_root}')
    
    def quit_app(self, e):
        """Quits the app!"""
        self.quit()
    
    def check_houghB_state(self, e):
        """Changes the state of hough transform button if pressed."""
        if self.is_slid:
            self.check_slidingB_state(e)
            
        if self.is_hough:
            self.hough_button_frame1.configure(fg_color=GRAY1)
            self.hough_button_lbl.configure(text_color="gray")
            self.is_hough = False
        else:
            self.hough_button_frame1.configure(fg_color=YELLOW)
            self.hough_button_lbl.configure(text_color="white")
            self.is_hough = True
            
    def check_slidingB_state(self, e):
        """changes the state of Sliding window button if pressed."""
        if self.is_hough:
            self.check_houghB_state(e)
            
        if self.is_slid:
            self.slid_wind_frame1.configure(fg_color=GRAY1)
            self.slid_wind_lbl.configure(text_color="gray")
            self.is_slid = False
        else:
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
    
# remove title bar
# add two buttons at the left for Hough transform and sliding window
# add another title list for parameters beneath them
# add video section
# add player section and start, pause
