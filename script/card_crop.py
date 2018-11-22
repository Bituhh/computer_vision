import cv2
import numpy as np

cam = cv2.VideoCapture(1)
min_area = 2000
max_area = 120000

def getContour(img, kernel_shape):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,11,17,17)
    blur = cv2.GaussianBlur(gray, kernel_shape, 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2, 2) 
    erode = cv2.erode(thresh, kernel, iterations = 1)
    canny = cv2.Canny(blur, blur.mean()*0.01, blur.mean()*1.5)
    img, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sqr_cnts = []
    corners = []   
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = 0.1*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,perimeter,True)
        if area > min_area and area < max_area and len(approx) is 4:        
            sqr_cnts.append(contour)
            corners.append(approx)
    return sqr_cnts, corners


while cam.isOpened():
    ret, frame = cam.read()
    contours, corners = getContour(frame, (9, 9))
    if len(corners) is not 0:
        approx = corners[-1]
        a_dis = abs(approx[0, 0, 0] - approx[1, 0, 0]) + abs(approx[0 , 0, 1] - approx[1, 0, 1])
        b_dis = abs(approx[1, 0, 0] - approx[2, 0, 0]) + abs(approx[1 , 0, 1] - approx[2, 0, 1])
        if a_dis > b_dis:
            pts1 = np.float32([approx[3, 0], approx[0, 0], approx[2, 0], approx[1, 0]])
        else:
            pts1 = np.float32([approx[0, 0], approx[1, 0], approx[3, 0], approx[2, 0]])
        pts2 = np.float32([[0,0],[400, 0], [0, 600], [400, 600]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (400, 600))
    cv2.imshow('show', result)
    cv2.imshow('contours', frame)
    #cv2.imshow('thresh', closing)
    #cv2.imshow('canny', erode)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()
