import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import pickle as pkl
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel



labels = ["Scissors",
"Scissors",
"Needle driver",
"Needle driver",
"Forceps",
"Forceps",
"Empty",
"Empty",
"None"]

orig_labels = ["Right Scissors",
"Left Scissors",
"Right Needle driver",
"Left Needle driver",
"Right Forceps",
"Left Forceps",
"Right Empty",
"Left Empty",
"Left None",
"Right None",
]


tool_usage ={ "T0": "Empty",
 "T1":"Needle driver" ,
 "T2": "Forceps",
 "T3": "Scissors"}

def calculate_metrics(pred, gt):
    # pred: list of predictions
    # gt: list of ground truth
    acc = 0
    recall =    [0, 0, 0, 0, 0, 0, 0, 0]
    precision = [0, 0, 0, 0, 0, 0, 0, 0]
    f1 =        [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(pred)):
        for j in range(2):
            if "None" in pred[i][j]:
                continue
            p = orig_labels.index(("Right " if j==0 else "Left ")+ pred[i][j])
            g = orig_labels.index(("Right " if j==0 else "Left ")+ gt[i][j])
            if p == g:
                acc += 1
            if p != 8:
                if p == g:
                    recall[p] += 1
                precision[p] += 1
    for i in range(8):
        if (len([x for x in gt if orig_labels.index("Right "+ x[0]) == i or orig_labels.index("Left "+ x[1]) == i])) == 0:
            recall[i] = 0
        else:
            recall[i] = recall[i] / ((len([x for x in gt if orig_labels.index("Right "+ x[0]) == i]))+(len([x for x in gt if orig_labels.index("Left "+ x[1]) == i])))
        if (len([x for x in pred if orig_labels.index("Right "+ x[0]) == i or orig_labels.index("Left "+ x[1]) == i])) == 0:
            precision[i] = 0
        else:
            precision[i] = precision[i] / ((len([x for x in pred if orig_labels.index("Right "+ x[0]) == i]))+(len([x for x in pred if orig_labels.index("Left "+ x[1]) == i])))
    acc = acc / (len(pred)*2)
    f1 = [2 * recall[i] * precision[i] / (recall[i] + precision[i]) if recall[i] + precision[i] != 0 else 0 for i in range(8)]
    f1_macro = list(filter(lambda x: x != 0, f1))
    f1_macro = sum(f1_macro) / len(f1_macro)
    return recall, precision, f1, acc, f1_macro

def detect(save_img=False):
    source, weights, imgsz, trace = opt.source, opt.weights, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir /  save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

    
    # Set Dataloader
    vid_path, vid_writer = None, None
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    detections = []
    
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
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

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        
        
        detections.append(pred)
    
    frames_predictions = []
    for pred in detections:
        # res = None
        right = pred[0][(pred[0][:, 5]%2 == 0)]
        left = pred[0][(pred[0][:, 5]%2 == 1)]
        if len(right) == 0:
            right = 8
        else:
            # res = right[np.argmax(right[:, 4])]
            right = int(right[np.argmax(right[:, 4])][5])
        if len(left) == 0:
            left = 8
        else:
            # if res == None:
            #     res = left[np.argmax(left[:, 4])]
            # else:
            #     res = torch.cat((res, left[np.argmax(left[:, 4])]))
            left = int(left[np.argmax(left[:, 4])][5])
            
        # if res != None:
        #     pred[0] = res
        frames_predictions.append([right, left])
        
    # Smooth predictions with majotity voting based on past predictions
    # iterate from last to first frame
    for j, f_p in enumerate(frames_predictions[:4:-1]):
        f_p[0] = np.argmax(np.bincount([frames_predictions[-i][0] for i in range(j+1,j+5)]))
        f_p[1] = np.argmax(np.bincount([frames_predictions[-i][1] for i in range(j+1,j+5)]))
        
    frames_predictions = [[labels[right], labels[left]] for right, left in frames_predictions]
    
    left_ground_truth = open(opt.gt_path + f'/tools_left/{source.split("/")[-1].split(".")[0]}.txt', 'r').readlines()
    left_gt = []
    for line in left_ground_truth:
        start, end, label = line.split(' ')
        left_gt = left_gt + [tool_usage[label.replace("\n","")]] * (int(end) - int(start) + 1)
    right_ground_truth = open(opt.gt_path + f'/tools_right/{source.split("/")[-1].split(".")[0]}.txt', 'r').readlines()
    right_gt = []
    for line in right_ground_truth:
        start, end, label = line.split(' ')
        right_gt = right_gt + [tool_usage[label.replace("\n","")]] * (int(end) - int(start) + 1)
    ground_truth = [[right, left] for right, left in zip(right_gt, left_gt)]
    
    
    # compute metrics
    recall, precision, f1, acc, f1_macro = calculate_metrics(frames_predictions, ground_truth)

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    for (path, img, im0s, vid_cap), pred, frame_pred, gt in zip(dataset, detections, frames_predictions, ground_truth):        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                
            #add frame prediction
            cv2.putText(im0, f"Right: {frame_pred[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(im0, f"Left: {frame_pred[1]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(im0, f"GT Right: {gt[0]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(im0, f"GT Left: {gt[1]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')            

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
                    vid_writer.write(im0)
        

    print(f'Done. ({time.time() - t0:.3f}s)')
    #print the evaluation metrics with 2 decimal places
    print(f"Evaluation metrics: \n Recall: {recall} \n Precision: {precision} \n F1: {f1} \n Accuracy: {acc} \n F1-macro: {f1_macro}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    parser.add_argument('--gt_path', type=str, default='/home/student/code/data/tool_usage', help='ground truth path')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
