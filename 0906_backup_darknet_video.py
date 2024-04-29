from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import numpy as np
import math

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="/home/detectionteam/Desktop/jaehan/base_darknet/project/forklift/backup/0831yolov3_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./project/forklift/yolov3.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./project/forklift/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=0.2,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    # video = cv2.VideoWriter(output_video, fourcc, fps, (416,416))
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox

    _height     = darknet_height
    _width      = darknet_width
    # _height     = 416
    # _width      = 416
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
        #                            interpolation=cv2.INTER_LINEAR)
        frame_resized = cv2.resize(frame_rgb, (416, 416),
                                   interpolation=cv2.INTER_LINEAR)
        # frame_queue.put(frame)

#undistortion
        k1, k2, k3 = 0.5, 0.2, 0.0
        k1, k2, k3 = -0.32, 0.1, 0
        img = frame_resized
        rows, cols = img.shape[:2]
        mapy, mapx = np.indices((rows, cols),dtype=np.float32)


        mapx = 2*mapx/(cols-1)-1
        mapy = 2*mapy/(rows-1)-1
        r, theta = cv2.cartToPolar(mapx, mapy)


        ru = r*(1+k1*(r**2) + k2*(r**4) + k3*(r**6)) 


        mapx, mapy = cv2.polarToCart(ru, theta)
        mapx = ((mapx + 1)*cols-1)/2
        mapy = ((mapy + 1)*rows-1)/2

        frame_resized = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
#undistortion

        frame_queue.put(frame_resized)

        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        # img_for_detect = darknet.make_image(416, 416, 3)

        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        #print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
    cap.release()


def point_set(label):
    detections = detections_queue.get()
    for label, confidence, bbox in detections:
        x, y, w, h = darknet.bbox4points(bbox)
        
    x = int(x)
    y = int(y)
    middle = (x, y)


def distance(a, b):
    result = math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))
    return result


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    # video = set_saved_video(cap, args.out_filename, (416, 416))
    # pmx = None
    # fmx = None
    # hmx = None
    pm = None, None
    fm = None, None
    hm = None, None
    hfprint = None
    fprint = None
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []

        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)



            if label == "Person":
                x, y, w, h = darknet.bbox4points(bbox)
                pmx = int(x)
                pmy = int(y)
                pm = (pmx, pmy)
            if label == "Forklift":
                x, y, w, h = darknet.bbox4points(bbox)
                fmx = int(x)
                fmy = int(y)
                fm = (fmx, fmy)
            if label == "HandForklift":
                x, y, w, h = darknet.bbox4points(bbox)
                hmx = int(x)
                hmy = int(y)
                hm = (hmx, hmy)
            
            if (pm != (None, None)) and (hm != (None, None)):
                point_dist_h = distance(pm, hm)
                # print(point_dist_h)
                # cv2.line(image, (10, 10),(100,100), (0, 255, 0), 1)
                print(point_dist_h)
                if point_dist_h < 200 and point_dist_h > 30:
                    cv2.line(image, pm ,hm, (0, 255, 0), 1)

                    hfprint = 1
                    pm = None, None
                    hm = None, None

            if (pm != (None, None)) and (fm != (None, None)):
                point_dist_f = distance(pm, fm)
                # print(point_dist_f)
                cv2.line(image, pm ,fm, (0, 255, 0), 1)
                # cv2.line(image, (pm, fm), (0, 255, 0), 1)
                if point_dist_f < 150:
                    fprint = 1
                    pm = None, None
                    fm = None, None

            
            if hfprint == 1:
                for i in range(1, 3):
                    cv2.putText(image, "hf_collision",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)                
                    hfprint = 0
            if fprint == 1:
                for i in range(1, 3):
                    cv2.putText(image, "f_collision",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)                
                    fprint = 0

            
                    


            # print(pm)
            # print(fm)
            # print(hm)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not args.dont_show:
                cv2.imshow('Inference', image)
            if args.out_filename is not None:
                video.write(image)
            if cv2.waitKey(fps) == 27:
                break
    cap.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    # darknet_width = 416
    # darknet_height = 416
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_width = 416
    # video_height = 416
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
