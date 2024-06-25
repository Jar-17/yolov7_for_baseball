from PyQt5 import QtCore 
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
# from PyQt5.QtCore import QThread, pyqtSignal

import time
import os


from UI_for_releasepoint import Ui_MainWindow
from video_controller import video_controller,video_controller2,video_controller3

class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.button_openfile.clicked.connect(self.open_file)   
        self.ui.button_openfile2.clicked.connect(self.open_file_video2) 
        self.ui.button_openfile3.clicked.connect(self.open_file_video3) 
        self.ui.button_process.clicked.connect(self.process)

    def process(self):
        
        command = ""
        if self.video_path and self.video_path2:
            
            command = f"python D://mediapipe/pose_detection/POSE_frontview.py --source1 {self.video_path} --source2 {self.video_path2}"
            print(command)
       
        os.system(command)
        # self.video_controller.videoplayer_state = "play"
        # self.video_controller2.videoplayer_state = "play"  

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi)") # start path        
        self.video_path = filename
        self.video_controller = video_controller(video_path=self.video_path,
                                                 ui=self.ui)
        self.ui.label_filepath.setText(f"{self.video_path}")
        self.ui.button_play.clicked.connect(self.video_controller.play) # connect to function()
        self.ui.button_stop.clicked.connect(self.video_controller.stop)
        self.ui.button_pause.clicked.connect(self.video_controller.pause)
        self.ui.button_forward.clicked.connect(self.video_controller.forward)
        self.ui.button_rewind.clicked.connect(self.video_controller.rewind)

    def open_file_video2(self):
        filename2, filetype2 = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi)") # start path        
        self.video_path2 = filename2
        self.video_controller2 = video_controller2(video_path=self.video_path2,
                                                 ui=self.ui)
        self.ui.label_filepath2.setText(f"{self.video_path2}")
        self.ui.button_play2.clicked.connect(self.video_controller2.play) # connect to function()
        self.ui.button_stop2.clicked.connect(self.video_controller2.stop)
        self.ui.button_pause2.clicked.connect(self.video_controller2.pause)
        self.ui.button_forward2.clicked.connect(self.video_controller2.forward)
        self.ui.button_rewind2.clicked.connect(self.video_controller2.rewind)
        
    def open_file_video3(self):
        # filename3, filetype3 = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi)") # start path        
        self.video_path3 = "D://mediapipe/result/combined.mp4"
        self.video_controller3 = video_controller3(video_path=self.video_path3,
                                                    ui=self.ui)
        self.ui.button_play3.clicked.connect(self.video_controller3.play) # connect to function()
        self.ui.button_stop3.clicked.connect(self.video_controller3.stop)
        self.ui.button_pause3.clicked.connect(self.video_controller3.pause)
        self.ui.button_forward3.clicked.connect(self.video_controller3.forward)
        self.ui.button_rewind3.clicked.connect(self.video_controller3.rewind)

