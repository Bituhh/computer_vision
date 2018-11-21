import cv2
import numpy as np

cam = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
min_area = 2000
max_area = 120000


while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2, 2)
    #canny = cv2.Canny(blur, blur.mean()*0.01, blur.mean()*1.5)
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    #cv2.drawContours(frame, contours, -1, (255, 0, 0), 3) 
    for cnt in contours:        
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
            if area > min_area and area < max_area and len(approx) is 4:    
                cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)


    #closing = cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel)
    
    cv2.imshow('contours', frame)
    cv2.imshow('thresh', thresh)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()
