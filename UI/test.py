from tkinter import *


root = Tk()
root.overrideredirect(True)
root.geometry("500x300")

def move_app(e):
    root.geometry(f'+{e.x_root}+{e.y_root}')

def quitter(e):
    root.quit()
    
# create fake titlebar
title_bar = Frame(root, bg="darkgreen", relief="raised", bd=1)
title_bar.pack(expand=1, fill=X)

# create a title bar name for the app
title_label = Label(title_bar, text="RoadMaster", bg="darkgreen", fg="white")
title_label.pack(side=LEFT, pady=5)

# create close button
close_label = Label(title_bar, text=" X  ", bg="darkgreen", fg="white", relief="sunken")
close_label.pack(side=RIGHT, pady=5)
close_label.bind("<Button-1>", quitter)

# bind the titlebar so that the app moves with the mouse
title_bar.bind("<B1-Motion>", move_app)


my_button = Button(root, text="Close", font=("Helvetica, 32"), command=root.quit)
my_button.pack(pady=100)
root.mainloop()