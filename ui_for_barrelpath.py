import tkinter as tk
import os
from tkinter.constants import *
from tkinter import filedialog
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import csv

import tkinter as tk
import os
from tkinter.constants import *
from tkinter import filedialog
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import csv


window = tk.Tk()
window.title('barrel_path')
w = window.winfo_screenwidth()
h = window.winfo_screenheight()
window.geometry("%dx%d" %(w,h))
window.resizable(True, True)

file_name_label = tk.Label(text="File Adress",font=("Times New Roman",15))
file_name_label.place(x=240, y=10)

outcome_label = tk.Label(text="outcome",font=("Times New Roman",15))
outcome_label.place(x=0.75*w, y=0.5*h)

lowest_bat_img_label = tk.Label(text="lowest_bat_img",font=("Times New Roman",15))
lowest_bat_img_label.place(x=0.75*w, y=0.9*h)


loadFile_en = tk.Entry(width=40)
loadFile_en.place(x=350, y=10)

p = tk.StringVar()
p.set("20")
proportion_en = tk.Entry(width = 20,textvariable=p)

proportion_en.place(x=350 , y=40)

f = tk.StringVar()
f.set("240")
framerate_en = tk.Entry(width=20,textvariable=f)

framerate_en.place(x=350 , y=70)

t = tk.StringVar()
t.set("0.5")
threshold_en = tk.Entry(width =20,textvariable=t)
threshold_en.place(x=350 , y=100)

#label
proportion_lb = tk.Label(text='Proportion' ,font=("Times New Roman",15))
proportion_lb.place(x=240 , y=40)

framerate_lb = tk.Label(text='Framerate' ,font=("Times New Roman",15))
framerate_lb.place(x=240 , y=70)

threshold_lb = tk.Label(text='Threshold',font=("Times New Roman",15))
threshold_lb.place(x=240,y=100)

barrel_speed_lb = tk.Label(text='Barrel speed',font=("Times New Roman",15))
barrel_speed_lb.place(x=0.55*w,y=0.05*h)

barrel_angle_lb = tk.Label(text='Barrel angle',font=("Times New Roman",15))
barrel_angle_lb.place(x=0.55*w,y=0.1*h)

swing_time_lb = tk.Label(text='Swing time',font=("Times New Roman",15))
swing_time_lb.place(x=0.55*w,y=0.15*h)

exit_speed_lb = tk.Label(text='Exit speed(ball)',font=("Times New Roman",15))
exit_speed_lb.place(x=0.55*w,y=0.2*h)

exit_angle_lb = tk.Label(text='Exit angle(ball)',font=("Times New Roman",15))
exit_angle_lb.place(x=0.55*w,y=0.25*h)

barrel_speed_lb_outcome = tk.Label(font=("Times New Roman",15))
barrel_speed_lb_outcome.place(x=0.7*w,y=0.05*h)

barrel_angle_lb_outcome = tk.Label(font=("Times New Roman",15))
barrel_angle_lb_outcome.place(x=0.7*w,y=0.1*h)

swing_time_lb_outcome = tk.Label(font=("Times New Roman",15))
swing_time_lb_outcome.place(x=0.7*w,y=0.15*h)

exit_speed_lb_outcome = tk.Label(font=("Times New Roman",15))
exit_speed_lb_outcome.place(x=0.7*w,y=0.2*h)

exit_angle_lb_outcome = tk.Label(font=("Times New Roman",15))
exit_angle_lb_outcome.place(x=0.7*w,y=0.25*h)

barrel_speed_lb_unit = tk.Label(text='km/hr',font=("Times New Roman",15))
barrel_speed_lb_unit.place(x=0.8*w,y=0.05*h)

barrel_angle_lb_unit = tk.Label(text='degrees',font=("Times New Roman",15))
barrel_angle_lb_unit.place(x=0.8*w,y=0.1*h)

swing_time_lb_unit = tk.Label(text='ms',font=("Times New Roman",15))
swing_time_lb_unit.place(x=0.8*w,y=0.15*h)

exit_speed_lb_unit = tk.Label(text='km/hr',font=("Times New Roman",15))
exit_speed_lb_unit.place(x=0.8*w,y=0.2*h)

exit_angle_lb_unit = tk.Label(text='degrees',font=("Times New Roman",15))
exit_angle_lb_unit.place(x=0.8*w,y=0.25*h)

def GetScreenCenter():
    root = tk.Tk()
    return root.winfo_screenwidth()//2,root.winfo_screenheight()//2

def AdaptSize(img):
    # 视频、图片过大直接1/2
    center_x, center_y = GetScreenCenter()
    img_h, img_w, _ = img.shape
    if img_h > center_y * 2 or img_w > center_x * 2:
        img = cv2.resize(img, (img_w // 2, img_h // 2))
    return img


def Play():
    cap = cv2.VideoCapture("C:/Users/fubon/yolov7/runs/outcome/outcome.mp4") 
  
    # Check if camera opened successfully 
    if (cap.isOpened()== False): 
        print("Error opening video file") 
    
    # Read until video is completed 
    while(cap.isOpened()): 
        
    # Capture frame-by-frame 
        ret, frame = cap.read() 
        frame = cv2.resize(frame,(1240,720))
        # center_x, center_y = GetScreenCenter()
        # img=AdaptSize(img)
        # img_h,img_w,_=img.shape
        # t_x, t_y = (center_x - img_w // 2), (center_y - img_h // 2)
        
        if ret == True: 
        # Display the resulting frame 
            cv2.imshow('video', frame) 
            # cv2.moveWindow('video', t_x, t_y)

        # Press Q on keyboard to exit 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    
    # Break the loop 
        else: 
            break
    
    # When everything done, release 
    # the video capture object 
    cap.release() 
    
    # Closes all the frames 
    cv2.destroyAllWindows()

def load_img():
    try:
        grounded_path = str("C:/Users/fubon/yolov7/path.png")
        img = Image.open(grounded_path)
        resized_image= img.resize((600,420), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(resized_image)
        return img
    except:
        pass

def refresh():
    global img
    update_outcome("C:/Users/fubon/yolov7/runs/outcome/outcome.mp4")
    img = load_img()
    image_canvas_outcome.create_image(0,0,anchor=NW,image = img)

def update_outcome(file_path):
    global cap_outcome, outcome_is_playing
    
    cap_outcome = None

    if cap_outcome is not None:
        cap_outcome.release()
    
    cap = cv2.VideoCapture(file_path)
    
    while(cap.isOpened()): 
      
# Capture frame-by-frame 
        ret, frame = cap.read() 
        if ret == True: 
        # Display the resulting frame 
            cv2.imshow('Frame', frame) 
            
        # Press Q on keyboard to exit 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    
    # Break the loop 
        else: 
            break
    
        # When everything done, release 
        # the video capture object 
        cap.release() 
        
        # Closes all the frames 
        cv2.destroyAllWindows()

    # outcome_is_playing = True
    
    # play_video_outcome()

def play_video_outcome(file_path):
    global outcome_is_playing
    
    # try:
    #     ret, frame = cap_outcome.read()
    # except:
    #     update_outcome(file_path)
    if ret and outcome_is_playing:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1080, 720))
        cv2.imshow("outcome", frame)
        cv2.imshow("outcome", frame)
        
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        cap_outcome.release()
        cv2.destroyAllWindows()
        
    else:
        play_video_outcome()
        if play_video_outcome(cv2.CAP_PROP_POS_FRAMES) == play_video_outcome(cv2.CAP_PROP_POS_FRAMES_COUNT):
            play_video_outcome.SET(CV2.CAP_PROP_POS_FRAMES, 0)
            play_video_outcome()
        # elif cap_outcome.GET(CV2.CAP_PROP_POS_FRAMES) == (cap_outcome.get(CV2.CAP_PROP_POS_FRAMES_COUNT)/2)   :


def loadFile():
    if loadFile_en.get() == "":
        file_path = filedialog.askopenfilename(initialdir="/examples")
        loadFile_en.insert(0, file_path)
        
        update_preview(file_path)
    else:
        file_path = filedialog.askopenfilename(initialdir="/examples")
        loadFile_en.delete(0, 'end')
        loadFile_en.insert(0, file_path)
        
        update_preview(file_path)



#preview
def update_preview(file_path):
    global cap, is_playing
    
    if cap is not None:
        cap.release()
    
    cap = cv2.VideoCapture(file_path)
    is_playing = True
    
    play_video()


    

#play
def play_video():
    global is_playing
    
    ret, frame = cap.read()
    if ret and is_playing:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (600, 420))
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        video_canvas.create_image(0, 0, image=photo, anchor='nw')
        video_canvas.image = photo
        window.after(10, play_video)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        play_video()
        
def play_video_imgshow():
    global is_playing
    
    ret, frame = cap.read()
    if ret and is_playing:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1080, 720))
        cv2.imshow("batting", frame)
        
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        play_video_imgshow()

def enter_data():
    command = ""
    #32 for ball , 34 for bat
    if loadFile_en.get():
        # if   check_var1.get() == 1: #--conf-thres 更改threshhold沒什麼幫助
        command = f"python detect_or_track_v2_922.py --conf-thres {threshold_en.get()} --weights yolov7_training.pt --no-trace --view-img --source {loadFile_en.get()} --seed 2 --track --classes 32 34  --show-track --nobbox --nolabel --proportion {proportion_en.get()} --framerate {framerate_en.get()} --video_format 1 "

        print(loadFile_en.get())
        print(command)
        # elif check_var2.get() == 1:
        #     if check_var3.get() == 1:
        #         command = f"python detect_or_track_0.5_v2_922.py --weights yolov7.pt --small 1 --no-trace --view-img --source {loadFile_en.get()} --seed 2 --track --classes 32 34  --show-track --nobbox --nolabel --proportion {proportion_en.get()} --framerate {framerate_en.get()} --video_format 2"
        #     else:
        #         command = f"python detect_or_track_0.5_v2_922.py --weights yolov7.pt --img-size 1024 --no-trace --view-img --source {loadFile_en.get()} --seed 2 --track --classes 32 34  --show-track --nobbox --nolabel --proportion {proportion_en.get()} --framerate {framerate_en.get()} --video_format 2"
        
        os.system(command)
        
        with open('output.csv', newline='') as csvfile:

        # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile)
            data = list(rows)   

        # print(data[0][1])
        # print(data[0][1].type)


        swing_time_lb_outcome['text'] = str(round(float(data[1][2]),2))
        barrel_speed_lb_outcome['text'] = str(round(float(data[1][0]),2))
        barrel_angle_lb_outcome['text'] = str(round(float(data[1][1]),2))
        exit_speed_lb_outcome['text'] = str(round(float(data[1][3]),2))
        exit_angle_lb_outcome['text'] = str(round(float(data[1][4]),2))
        
    else:
        messagebox.showwarning(title="Cannot find the file", message="Please choose the right")

load = tk.Button(text="load file", font=("Times New Roman",10) , command=loadFile)
load.place(x=600, y=5)

#execute button

show_video_button = tk.Button(text="Show video", font=("Times New Roman",12), command=Play)
show_video_button.place(x=0.45*w, y=0.35*h)

execute_button = tk.Button(text="Analyze", font=("Times New Roman",12), command=enter_data)
execute_button.place(x=0.45*w, y=0.45*h)

# #checkbutton
# check_var1 = tk.IntVar()
# checkbox1 = tk.Checkbutton(text="normal", variable=check_var1, anchor='center', font=("Helvetica", 16))
# checkbox1.place(x=0.1*w, y=0.8*h)

# check_var3 = tk.IntVar()
# checkbox3 = tk.Checkbutton(text="300 fps", variable=check_var3, anchor='center', font=("Helvetica", 16))
# checkbox3.place(x=0.25*w, y=0.8*h)

# check_var2 = tk.IntVar()
# checkbox2 = tk.Checkbutton(text="slow", variable=check_var2, anchor='center', font=("Helvetica", 16))
# checkbox2.place(x=0.4*w, y=0.8*h)

refresh_button = tk.Button(text="Show", font=("Times New Roman",12) , command=refresh)
refresh_button.place(x=0.45*w, y=0.55*h)


video_canvas = tk.Canvas(window, width=640, height=460)
# video_canvas.pack(fill="both",expand=True)
video_canvas.place(x=0.005*w, y=0.15*h)

# video_canvas_outcome = tk.Canvas(window, width=600, height=420)
# video_canvas_outcome.place(x=0.55*w, y=0.005*h)

image_canvas_outcome = tk.Canvas(window, width=600, height=420)
image_canvas_outcome.place(x=0.55*w, y=0.45*h)



cap = None
is_playing = False

window.mainloop()