import cv2
import numpy as np

cam = cv2.VideoCapture(1)
min_area = 2000
max_area = 120000

def getFeatures(img, kernel_shape):
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
        corner = cv2.approxPolyDP(contour,perimeter,True)
        if area > min_area and area < max_area and len(corner) is 4:        
            sqr_cnts.append(contour)
            corners.append(corner)
    return sqr_cnts, corners

def extractCard(corner):
    a_dis = abs(corner[0, 0, 0] - corner[1, 0, 0]) + abs(corner[0 , 0, 1] - corner[1, 0, 1])
    b_dis = abs(corner[1, 0, 0] - corner[2, 0, 0]) + abs(corner[1 , 0, 1] - corner[2, 0, 1])
    if a_dis > b_dis:
        pts1 = np.float32([corner[3, 0], corner[0, 0], corner[2, 0], corner[1, 0]])
    else:
        pts1 = np.float32([corner[0, 0], corner[1, 0], corner[3, 0], corner[2, 0]])
    pts2 = np.float32([[0,0],[200, 0], [0, 300], [200, 300]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (200, 300))

save = False
label = 0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while cam.isOpened():
    ret, img = cam.read()
    contours, corners = getFeatures(img, (9, 9))
    cv2.drawContours(img, contours, -1, (255, 0 ,0), 3)
    if len(corners) is not 0:
        corner = corners[-1]
        card = extractCard(corner)
        card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        card_gray = cv2.bilateralFilter(card_gray,11,17,17)
        card_thresh = cv2.adaptiveThreshold(card_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2, 2)
        #card_opening = cv2.morphologyEx(card_thresh, cv2.MORPH_OPEN, kernel)

        cv2.imshow('card', card)
        cv2.imshow('thresh', card_thresh[5:75, 5:35])
    cv2.imshow('camera', img)
    
    if save is True and label is not 50:
        cv2.imwrite('../images/' + str(label) + '-' + card_name + '.png', card_thresh[5:75, 5:35])
        label = label + 1
    else:
        save = False
        label = 0
        
    if label is 49:
        print('ready for next card!')
        
    if cv2.waitKey(1) & 0xFF == ord('a'):
        card_name = str(input('Which card to save: '))
        save = True

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()
