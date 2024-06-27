from PyQt5 import QtCore 
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
# from PyQt5.QtCore import QThread, pyqtSignal

import time
import os
import cv2
import csv

from UI_for_barrelpath import Ui_MainWindow
from video_controller import video_controller,video_controller2,video_controller3

class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.button_openfile.clicked.connect(self.open_file)   
        # self.ui.button_openfile2.clicked.connect(self.open_file_video2) 
        self.ui.button_openfile3.clicked.connect(self.open_file_video3) 
        self.ui.button_process.clicked.connect(self.process)
        self.ui.button_showimg.clicked.connect(self.showimg)

    def showimg(self):
        img_path = "C:/Users/fubon/yolov7/path.png"
        img = cv2.imread(img_path)
        img = cv2.resize(img,(960,540))
        cv2.namedWindow("barrel and ball path",0)
        cv2.resizeWindow("barrel and ball path", 960, 540)
        cv2.imshow("barrel and ball path",img)


    def process(self):
        
        command = ""
        if self.video_path :
            
            command = f"python C:/Users/fubon/yolov7/detect_or_track_v2_922.py --conf-thres 0.5 --weights yolov7_training.pt --no-trace --view-img --source {self.video_path} --seed 2 --track --classes 32 34  --show-track --nobbox --nolabel --proportion 15 --framerate 240 --video_format 1 "

            print(command)
       
        os.system(command)

        with open('output.csv', newline='') as csvfile:

        # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile)
            data = list(rows)   

        # print(data[0][1])
        # print(data[0][1].type)


        self.ui.swing_time_lb_outcome.setText(str(round(float(data[1][2]))))
        self.ui.bat_speed_lb_outcome.setText(str(round(float(data[1][0]))))
        self.ui.swing_angle_lb_outcome.setText(str(round(float(data[1][1]))))
        self.ui.ball_exit_speed_lb_outcome.setText(str(round(float(data[1][3]))))
        self.ui.ball_exit_angle_lb_outcome.setText(str(round(float(data[1][4]))))
        self.ui.projected_dis_lb_outcome.setText(str(round(float(data[1][5]))))
        # self.video_controller.videoplayer_state = "play"
        # self.video_controller2.videoplayer_state = "play"  

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi *.mov)") # start path        
        self.video_path = filename
        self.video_controller = video_controller(video_path=self.video_path,
                                                 ui=self.ui)
        self.ui.label_filepath.setText(f"{self.video_path}")
        self.ui.button_play.clicked.connect(self.video_controller.play) # connect to function()
        self.ui.button_stop.clicked.connect(self.video_controller.stop)
        self.ui.button_pause.clicked.connect(self.video_controller.pause)
        self.ui.button_forward.clicked.connect(self.video_controller.forward)
        self.ui.button_rewind.clicked.connect(self.video_controller.rewind)

    # def open_file_video2(self):
    #     filename2, filetype2 = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi)") # start path        
    #     self.video_path2 = filename2
    #     self.video_controller2 = video_controller2(video_path=self.video_path2,
    #                                              ui=self.ui)
    #     self.ui.label_filepath2.setText(f"{self.video_path2}")
    #     self.ui.button_play2.clicked.connect(self.video_controller2.play) # connect to function()
    #     self.ui.button_stop2.clicked.connect(self.video_controller2.stop)
    #     self.ui.button_pause2.clicked.connect(self.video_controller2.pause)
    #     self.ui.button_forward2.clicked.connect(self.video_controller2.forward)
    #     self.ui.button_rewind2.clicked.connect(self.video_controller2.rewind)
        
    def open_file_video3(self):
        # filename3, filetype3 = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi)") # start path        
        self.video_path3 = "C:/Users/fubon/yolov7/runs/outcome/outcome.mp4"
        self.video_controller3 = video_controller3(video_path=self.video_path3,
                                                    ui=self.ui)
        self.ui.button_play3.clicked.connect(self.video_controller3.play) # connect to function()
        self.ui.button_stop3.clicked.connect(self.video_controller3.stop)
        self.ui.button_pause3.clicked.connect(self.video_controller3.pause)
        self.ui.button_forward3.clicked.connect(self.video_controller3.forward)
        self.ui.button_rewind3.clicked.connect(self.video_controller3.rewind)

