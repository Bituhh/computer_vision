import cv2
import numpy as np

cam = cv2.VideoCapture(0)

def getContour(img, kernel_shape):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,11,17,17)
    blur = cv2.GaussianBlur(gray, kernel_shape, 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2, 2)
    erode = cv2.erode(thresh, kernel, iterations = 1)
    canny = cv2.Canny(blur, blur.mean()*0.01, blur.mean()*1.5)
    img, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

if __name__ == '__main__':
    
    while cam.isOpened():
        ret, frame = cam.read()
        contours = getContour(frame, (3, 3))
        cv2.drawContours(frame, contours, -1, (255, 0 ,0), 3)
        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cam.release()
    cv2.destroyAllWindows()


