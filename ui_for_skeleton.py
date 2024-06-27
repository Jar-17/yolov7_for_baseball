import tkinter as tk
import os
from tkinter.constants import *
from tkinter import filedialog,Label,Tk
from tkinter import messagebox
from tkinter import *
import cv2
from PIL import Image, ImageTk

window = tk.Tk()
window.title('skeleton')
window.geometry('900x720')
window.resizable(True, True)

file_name_label = tk.Label(text="File Adress:")
file_name_label.place(x=240, y=10)

outcome_label = tk.Label(text="outcome")
outcome_label.place(x=180, y=650)

grounded_img_label = tk.Label(text="grounded_img")
grounded_img_label.place(x=600, y=650)

loadFile_en = tk.Entry(width=40)
loadFile_en.place(x=340, y=10)

def load_img():
    try:
        grounded_path = str("D:/mediapipe/result/opt.source.mp4.jpg")
        img = Image.open(grounded_path)
        resized_image= img.resize((320,240), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(resized_image)
        return img
    except:
        pass


def refresh():
    global img
    update_outcome("D:/mediapipe/result/opt.source.mp4")
    img = load_img()
    image_canvas_outcome.create_image(0,0,anchor=NW,image = img)


def loadFile():
    if loadFile_en.get() == "":
        file_path = filedialog.askopenfilename(initialdir="/mediapipe/pose_detection")
        loadFile_en.insert(0, file_path)
        
        update_preview(file_path)

        # try:
        #     update_outcome("D:/mediapipe/resultopt.source.mp4")
        #     image_canvas_outcome.create_image(0,0,anchor=NW,image = img)

        # except:
        #     pass

        
        

    else:
        file_path = filedialog.askopenfilename(initialdir="/mediapipe/pose_detection")
        loadFile_en.delete(0, 'end')
        loadFile_en.insert(0, file_path)
        
        update_preview(file_path)

#         try:
#             update_outcome("D:/mediapipe/resultopt.source.mp4")
#             image_canvas_outcome.create_image(0,0,anchor=NW,image = img)

#         except:
#             pass
# #preview
def update_preview(file_path):
    global cap_preview, preview_is_playing
    
    cap_preview = None

    if cap_preview is not None:
        cap_preview.release()
    
    cap_preview = cv2.VideoCapture(file_path)

    preview_is_playing = True
    
    play_video_preview()

def update_outcome(file_path):
    global cap_outcome, outcome_is_playing
    
    cap_outcome = None

    if cap_outcome is not None:
        cap_outcome.release()
    
    cap_outcome = cv2.VideoCapture(file_path)
    outcome_is_playing = True
    
    play_video_outcome()

#play
def play_video_preview():
    global preview_is_playing
    
    ret, frame = cap_preview.read()
    if ret and preview_is_playing:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 300))
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        
        video_canvas_preview.create_image(0, 0, image=photo, anchor='nw')
        video_canvas_preview.image = photo
        window.after(10, play_video_preview)
    else:
        cap_preview.set(cv2.CAP_PROP_POS_FRAMES, 0)
        play_video_preview()

def play_video_outcome():
    global outcome_is_playing
    
    ret, frame = cap_outcome.read()
    if ret and outcome_is_playing:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 300))
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        
        video_canvas_outcome.create_image(0, 0, image=photo, anchor='nw')
        video_canvas_outcome.image = photo
        window.after(10, play_video_outcome)
    else:
        cap_outcome.set(cv2.CAP_PROP_POS_FRAMES, 0)
        play_video_outcome()

def enter_data():
    command = ""
    if loadFile_en.get():
        
        command = f"python POSE.py --source {loadFile_en.get()}"
       
        os.system(command)
    else:
        messagebox.showwarning(title="Cannot find the file", message="Please choose the right")


load = tk.Button(text="load file", command=loadFile)
load.place(x=600, y=5)

#execute button
execute_button = tk.Button(text="Execute Command", command=enter_data)
execute_button.place(x=390, y=370)

refresh_button = tk.Button(text="refresh", command=refresh)
refresh_button.place(x=390, y=650)


video_canvas_preview = tk.Canvas(window, width=320, height=240)
video_canvas_preview.place(x=310, y=100)

video_canvas_outcome = tk.Canvas(window, width=320, height=240)
video_canvas_outcome.place(x=100, y=400)

image_canvas_outcome = tk.Canvas(window, width=320, height=240)
image_canvas_outcome.place(x=500, y=400)



cap = None
is_playing = False

window.mainloop()

