import argparse
# from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import time
import json
# from utils.general import increment_path
#import skvideo.io
#import serial

#cam = cv2.VideoCapture(3)

mppose = mp.solutions.pose
mpdraw = mp.solutions.drawing_utils
poses = mppose.Pose()
h = 0
w = 0

#ser = serial.Serial("COM3", 9600)

start_time = 0
status = False
throw = False

class DrawingSpecline:
    color:tuple[int , int , int] = (0,155,232)
    thickness: int = 2
    circle_radius : int = 10

class DrawingSpecdot:
    color:tuple[int , int , int] = (255,191,0)
    thickness: int = 2
    circle_radius : int = 1

def calc_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

def get_landmark(landmarks, part_name):
    return [
        landmarks[mppose.PoseLandmark[part_name].value].x,
        landmarks[mppose.PoseLandmark[part_name].value].y,
        landmarks[mppose.PoseLandmark[part_name].value].z,
    ]

def get_knee_angle(landmarks):
    

    r_hip = get_landmark(landmarks, "RIGHT_HIP")
    l_hip = get_landmark(landmarks, "LEFT_HIP")

    r_knee = get_landmark(landmarks, "RIGHT_KNEE")
    l_knee = get_landmark(landmarks, "LEFT_KNEE")

    r_ankle = get_landmark(landmarks, "RIGHT_ANKLE")
    l_ankle = get_landmark(landmarks, "LEFT_ANKLE")

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)

    m_hip = (r_hip + l_hip)
    m_hip = [x / 2 for x in m_hip]
    m_knee = (r_knee + l_knee)
    m_knee = [x / 2 for x in m_knee]
    m_ankle = (r_ankle + l_ankle)
    m_ankle = [x / 2 for x in m_ankle]

    mid_angle = calc_angles(m_hip, m_knee, m_ankle)

    return [int(r_angle),int(l_angle) ,int(mid_angle)]

def main():
    global h, w, start_time, status,frmae_count,r_wrist_arr,grounded
    frame_count = 0
    grounded = 0
    flag = False
    #set video 
    vid_path , vid_writer = None , None
    
    cam = cv2.VideoCapture(opt.source)

    if not cam.isOpened():
        print("Camera not open")
        exit()

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while not flag:
        
        save_path = "D:/mediapipe/result/"+"opt.source"+".mp4"

        ret, frame = cam.read()
        fps = cam.get(cv2.CAP_PROP_FPS)
        if not ret:
            print("Read Error")
            break

        rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        poseoutput = poses.process(rgbframe)
        h, w, _ = frame.shape
        preview = frame.copy()


        if poseoutput.pose_landmarks:
            mpdraw.draw_landmarks(preview, poseoutput.pose_landmarks, mppose.POSE_CONNECTIONS , DrawingSpecdot , DrawingSpecline)
            knee_angles = get_knee_angle(poseoutput.pose_landmarks.landmark)

        
            throwing_leg_angle = knee_angles[0]

            if throwing_leg_angle >= 175:
                throw = False        
            elif throwing_leg_angle <175:
                throw = True


            if vid_path != save_path: #new video
                vid_path = save_path
                test_path = save_path + ".jpg"
                if isinstance(vid_writer , cv2.VideoWriter):
                    vid_writer.release() #release previous video writer
                    
                vid_writer = cv2.VideoWriter(save_path , cv2.VideoWriter_fourcc(*'mp4v') , fps , (w,h))
        
            #draw the path of hand
        

            if frame_count==0:
                r_wrist = get_landmark(poseoutput.pose_landmarks.landmark, "RIGHT_WRIST")
                # r_wrist_pre = [0 for i in 1]
                # r_wrist_pre[0] = r_wrist  
                r_wrist_arr = [0 for i in range(1)]
                r_wrist_arr[0] = r_wrist

                left_foot = get_landmark(poseoutput.pose_landmarks.landmark, "LEFT_FOOT_INDEX") 
                lowest_left_foot = left_foot                
            else:
                

                r_wrist = get_landmark(poseoutput.pose_landmarks.landmark, "RIGHT_WRIST")
                
                r_wrist_arr.append(r_wrist) 
                # cv2.putText(preview, "FPS: " + str(r_wrist_arr[frame_count][0]*w), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
                # cv2.putText(preview, "FPS: " + str((1-r_wrist_arr[frame_count][1])*h), (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
                
                #preview = cv2.puttext(preview,'speed: %d km/hr'% 0,(20,100),(255,255,255),20)                
                [cv2.line(preview,(int(r_wrist_arr[i][0]*w),int(r_wrist_arr[i][1]*h)),(int(r_wrist_arr[i+1][0]*w),int(r_wrist_arr[i+1][1]*h)),(255,255,255), 2)   
                        for i in range(frame_count)
                            ]

                left_foot = get_landmark(poseoutput.pose_landmarks.landmark, "LEFT_FOOT_INDEX") 

                if left_foot[1]>lowest_left_foot[1]:
                    lowest_left_foot[1] = left_foot[1]
                    grounded_img = preview
                
                
                    
                    

            
            

                

                

        frame_count = frame_count + 1

        if throw:
            for _ in range(4):
                vid_writer.write(preview)
                cv2.imshow('frame', preview)
        else:
            vid_writer.write(preview)
            cv2.imshow('frame', preview)


        if cv2.waitKey(1) & 0xFF == ord('q'):

            flag = True

    cv2.imwrite(test_path, grounded_img)


    # release camera
    cam.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='result', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    

    opt = parser.parse_args()
    print(opt)
    


    main()