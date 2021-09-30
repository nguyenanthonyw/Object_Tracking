import cv2

cap = cv2.VideoCapture("highway.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2() #detect objects

while True:
    ret, frame = cap.read()

    mask = object_detector.apply(frame)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27: # Enter key
        break

cap.release()

