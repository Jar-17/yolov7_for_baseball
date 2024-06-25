import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import math
from numpy import random
from PIL import Image,ImageDraw,ImageFont
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from sort import *


def cv2AddChineseText(img , text , position , textColor , textSize):
    if (isinstance(img , np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img , cv2.COLOR_BGR2RGB))

        #create a target to draw
        draw = ImageDraw.Draw(img)
        #font size
        fontStyle = ImageFont.truetype('NotoSansTC-VariableFont_wght.ttf' , textSize , encoding="utf-8")
        #draw it
        draw.text(position , text , textColor , font=fontStyle)
        #return the form of cv2
        return cv2.cvtColor(np.asarray(img) , cv2.COLOR_RGB2BGR)

"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]
        
        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            label = str(id) + ":"+ names[cat] if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    return img


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if not opt.nosave:  
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride) 
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)  #the video is transform to imgs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    ###################################
    global swinging , frame_num , swing_time ,max_dis
    
    frame_num = 0
    startTime = 0
    slope = 0
    swing_time=0
    max_dis=0
    ###################################
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)                                                                                      #??
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference (this is already detection , and pred record which frame has detect something)
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()
        frame_num = len(im0s)

        # Apply NMS (to filter out redundent bounding box)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        total_time = 0
        swinging = False

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            shadow_color=(255,255,0)

            
            if len(det): #how many items are detected
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0,6))
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))

                opt.track = True

                if opt.track:
  
                    tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                    tracks =sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        if opt.show_track:
                            #loop over tracks
                            for t, track in enumerate(tracks):
                  
                               #detclass related to that in coco      
                               #deal with one kind of track at a time

                                track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]


                                im0_height, im0_width, _ = im0.shape
                                shadow_color=(255,255,0)
                                
                                #i is the former one , i+1 is the latter one
                                
                                [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                int(track.centroidarr[i][1])), 
                                                (int(track.centroidarr[i+1][0]),
                                                int(track.centroidarr[i+1][1])),
                                                track_color, thickness=opt.thickness) 
                                                for i,_ in  enumerate(track.centroidarr) 
                                                    if i < len(track.centroidarr)-1 ] 
                                
                    global dis
                                
                    swinging = False

                                
                    dis = [0 for i in enumerate(track.centroidarr)] #initial an empty array
                                
                    for i,_ in enumerate(track.centroidarr):
                        if i < len(track.centroidarr)-1:
                            dis[i] = (((int(track.centroidarr[i][0])-int(track.centroidarr[i+1][0]))**2+(int(track.centroidarr[i][1])-int(track.centroidarr[i+1][1]))**2)**(1/2))*opt.proportion*opt.framerate*0.036
                           
                    sorted_dis = sorted(dis , reverse = True , key=lambda x: x.imag) #with the biggest in th front      

                    if i>10:
                        attack_angle = sorted(track.centroidarr , reverse = False , key=lambda x:x[1] )            
                        slope_part = -1*math.atan(attack_angle[10][1]-attack_angle[0][1])/(attack_angle[10][0]-attack_angle[0][0])*180/math.pi 
                                        
                        #prevent the out range 
                        if slope_part > slope :
                            if slope_part < 100:
                                slope = slope_part

                    

                    for j in dis:
                        if j> 20*opt.proportion*opt.framerate*0.036:
                            swing_time+=1
                            swinging = True
                    total_time = swing_time/opt.framerate

                    if sorted_dis[0] > max_dis:
                        max_dis = sorted_dis[0]                
                                
                         
                                
                else:
                    bbox_xyxy = dets_to_sort[:,:4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]
                
                #draw all the detection at once
                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)

            
                    
                
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            ######################################################
            if dataset.mode != 'image' and opt.show_fps:
                currentTime = time.time()

                fps = 1/(currentTime - startTime)
                startTime = currentTime
                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)

            #######################################################
            

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    
                    #[cv2.putText(im0,'barrel speed: %.2f' % max_dis,(20, 110) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, shadow_color , 1)]
                    #[cv2.putText(im0,'swing time: %.4f' % total_time,(20, 140) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, shadow_color , 1)]
                    #在第10個frame
                    #if i == len(pred)-10:
                    #    attack_angle = sorted(track.centroidarr , reverse = False , key=lambda x:x[1] )
                    #    slope = (attack_angle[10][1]-attack_angle[0][1])/(attack_angle[10][0]-attack_angle[0][0])

                    if opt.small==0:
                        im0 = cv2AddChineseText(im0,'揮棒速度: %d km/hr'% max_dis,(20,100),shadow_color,20)
                        if i > len(pred)-10:
                            im0 = cv2AddChineseText(im0,"揮棒角度: %.2f" % slope,(20,130),shadow_color,20)
                        im0 = cv2AddChineseText(im0,"揮棒時間: %.3f ms" % total_time,(20,160),shadow_color,20)
                    else:
                        im0 = cv2AddChineseText(im0,'揮棒速度: %d km/hr'% max_dis,(2,2),shadow_color,10)
                        if i > len(pred)-10:
                            im0 = cv2AddChineseText(im0,"揮棒角度: %.2f" % slope,(2,14),shadow_color,10)
                        im0 = cv2AddChineseText(im0,"揮棒時間: %.3f ms" % total_time,(2,26),shadow_color,10)    

                    #1
                    if opt.video_format == 1:
                        vid_writer.write(im0)
                
                    #2
                    if opt.video_format == 2:
                        #frames = vid_writer.get_frames()
                        
                        if swinging:
                            for _ in range(4):  # Repeat the entire video sequence twice
                        #    for im in frames:
                                vid_writer.write(im0)
                        else:
                            vid_writer.write(im0)

                    #3
                    #if opt.video_format == 3:


                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')
    parser.add_argument('--proportion' , type = float , default=1 )
    parser.add_argument('--framerate' , type=float , default=1 )
    parser.add_argument('--video_format' , type=int , default=1 )
    parser.add_argument('--small' , type=int , default=0 )

    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5,
                       min_hits=2,
                       iou_threshold=0.2) 

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect() #so the detection doesnt loop

    
    #plt.show()

