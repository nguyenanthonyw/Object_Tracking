import cv2
from tracker import *


# This program takes a video file and track how many objects pass on the road

cap = cv2.VideoCapture("highway.mp4")


tracker = EuclideanDistTracker()

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40 ) #detect objects

while True:
    ret, frame = cap.read()


    height, width, _ = frame.shape


    roi = frame [340: 720, 500:800]




    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        #cv2.drawContours(roi, [cnt], -1, (0, 255,0) ,2 )

        area = cv2.contourArea(cnt)
        if area > 100:
            x, y,  w, h = cv2.boundingRect(cnt)

            detections.append([x, y,w,h])

    boxes_ids = tracker.update(detections)
    print(boxes_ids)

    for boxes_ids in boxes_ids:
        x, y, w, h, id = boxes_ids
        cv2.putText(roi, str(id), (x ,y - 15), cv2.FONT_HERSHEY_PLAIN, 1 , (255, 0,0),2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)



    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27: # Enter key
        break

cap.release()

