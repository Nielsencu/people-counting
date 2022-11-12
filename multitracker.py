# import the necessary packages
import numpy as np
import cv2
import time
import torch
from sort.sort import *
from imutils.object_detection import non_max_suppression

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1, y1, x2, y2 = boxA
    x3, y3, x4, y4 = boxB
    x_inter1 = max(x1,x3)
    y_inter1 = max(y1,y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2-x1)
    height_box1 = abs(y2-y1)
    width_box2 = abs(x4-x3)
    height_box2 = abs(y4-y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    min_area = min(area_box1, area_box2)
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / min_area
    return iou

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Set up tracker.
# Instead of CSRT, you can also use

#create instance of SORT
tracker = Sort(max_age=1, min_hits=10, iou_threshold=0.15) 

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

persons = []
if not cap.isOpened():
    print("Failed")

FIND = 0
TRACK = 1
state = FIND
count = 0
ids = dict()
gone = 0
last_detected = time.time()
last_tracked = time.time()
inFrameCount = 0
inFrameCounts = []
idsInFrame = dict()
yolo = 1
if yolo:
    # Load the model from torch.hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
    model.conf = 0.35  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.classes = [0]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    model.amp = True  # Automatic Mixed Precision (AMP) inference
    detect = lambda frame : model(frame, size=640)
else:
    detect = lambda frame : hog.detectMultiScale(frame, winStride=(8,8) ,padding=(8,8),scale=1.05)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if yolo:
        results = detect(frame)
        if results.xyxy:
            boxes = results.pandas().xyxy[0].to_numpy()
            boxes = np.array([box for box in boxes if box[-1] == "person"])
    else:
        boxes, _ = detect(frame)

    if len(np.shape(boxes)) == 2:
        box_count = 0
        print(f"Detected {np.shape(boxes)[0]} boxes")
        for (xA, yA, xB, yB) in [[x, y, x + w, y + h] for (x, y, w, h) in boxes[:,:4].astype(int)]:
            last_detected = time.time()
            box_count+=1
            cv2.putText(frame, f"Box {box_count}", (xA-10, yB+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB),
                            (0, 255, 0), 2)

    if state == FIND:
        # detect people in the image
        # returns the bounding boxes for the detected objects
        if len(boxes) > 0:
            state = TRACK
        else:
            cv2.putText(frame, "No persons detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    elif state == TRACK:
        ok = False
        if len(boxes) > 0:
            # update SORT
            boxes = non_max_suppression(boxes[:,:4])
            tracks_bbs_ids = tracker.update(boxes)
            ok = len(tracks_bbs_ids) > 0
            if ok:
                last_tracked = time.time()
                gone = 0
        else:
            tracks_bbs_ids = tracker.update()
            print("Empty tracking")
            #cv2.putText(frame, "No persons detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        detectedIds = {trk.id : 1 for trk in tracker.getTrackers()}
        keys = tuple(detectedIds.keys())
        print(f"Detected ids in frame: {keys}")
        print(f"Ids was in frame: {idsInFrame}")

        copied_ids = idsInFrame.copy()

        for id in copied_ids:
            exist = detectedIds.get(id)
            if exist is None:
                idsInFrame[id]  += 1
                missed_count = idsInFrame[id]
                if missed_count > 3:
                    del idsInFrame[id]
                    inFrameCount -= 1

        for id in detectedIds:
            if id not in idsInFrame:
                print(f"New id appeared in frame {id}")
                inFrameCount+=1
                count += 1
                print(f"Count increased to {count}")
                idsInFrame[id] = 0

        # Draw bounding box
        if ok:
            box_count=0
            for bbox in tracks_bbs_ids:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                # rects.append(p1 + p2)
                # if len(rects) >= 2:
                #     score = iou(rects[0], rects[1])
                #     print(f"Iou is {score}")
                #     rects = np.array(rects)
                #     if score < 0.7:
                #         print("Iou smaller")
                #         gone +=1
                #     else:
                #         gone = 0
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                cv2.putText(frame, f"Box {box_count}", (p1[0]-10, p2[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                box_count+=1

            if len(tracks_bbs_ids) < inFrameCount:
                gone +=1
                inFrameCounts.append(tracks_bbs_ids)

        cv2.putText(frame, f"Current frame has {inFrameCount}, total : {count}", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        if time.time() - last_detected > 0.1:
            last_detected = time.time()
            print("No detections")
        if time.time() - last_tracked > 0.1:
            last_tracked = time.time()
            print("No tracking")

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)