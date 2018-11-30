import cv2
import numpy as np
from card_utils import Cards

# This code allow you to extract the suit and rank from a image.
# This was used to build a dataset from scratch. Image is extracted as thresholded
# images and saves in ../images folder. with label in the format of 6-4d (index-RankSuit)

if __name__ == '__main__':i
    cam = cv2.VideoCapture(0)
    save = False
    label = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while cam.isOpened():
        ret, img = cam.read()
        cards = Cards(img)
        corners, contours, hierarchy = cards.getFeatures((9, 9))
        cv2.drawContours(img, contours, -1, (255, 0 ,0), 3)
        for corner in corners:
            card = cards.extractCard(corner)
            card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
            card_gray = cv2.bilateralFilter(card_gray,11,17,17)
            card_thresh = cv2.adaptiveThreshold(card_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2, 2)
            cv2.imshow('card', card)
            cv2.imshow('Rank & Suit', card_thresh[7:77, 5:35])
        cv2.imshow('Camera Image', img)
        
        if save is True and label is not 250:
            cv2.imwrite('../images/' + str(label) + '-' + card_name + '.png', card_thresh[7:77, 5:35])
            label = label + 1
        else:
            save = False
            label = 0
            
        if label is 249:
            print('ready for next card!')
            
        # By pressing 'o' and option menu will prompt. 
        # 'c' will crop image and save to ../image folder.
        # anything else will exit the program.
        if cv2.waitKey(1) & 0xFF == ord('o'):
            if input('what do you wish to do? ') == 'p':
                card_name = str(input('Which card to save: '))
                save = True       
            else:
                break

    cam.release()
    cv2.destroyAllWindows()
