import cv2
import numpy as np

vid = cv2.VideoCapture('traffic.avi')
backSub = cv2.createBackgroundSubtractorMOG2()
counter = 0

while True:
    ret, frame = vid.read()
    if ret is False:
        break

    foregroundMask = backSub.apply(frame)

    cv2.line(frame, (50, 0), (50, 300), (255, 0, 0), 2)
    cv2.line(frame, (70, 0), (70, 300), (255, 0, 0), 2)

    contours, hierarchy = cv2.findContours(foregroundMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []

    for contour, hr in zip(contours, hierarchy):
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 40 and h > 40 and w < 115 and h < 155:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if x>50 and x<70:
                counter = counter + 1

    cv2.putText(frame, "Car : " + str(counter), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("video", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()


