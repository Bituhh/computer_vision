import cv2
#import numpy as np

cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('gray', gray)
    
    ret, thresh = cv2.threshold(gray, 50, 155, cv2.THRESH_BINARY)
    
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3) 
    cv2.imshow('threshold', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()
