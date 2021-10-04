import cv2

cap = cv2.VideoCapture("highway.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2() #detect objects

while True:
    ret, frame = cap.read()

    mask = object_detector.apply(frame)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(frame, [cnt], -1, (0, 255,0) ,2 )

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27: # Enter key
        break

cap.release()

