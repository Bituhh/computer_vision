from card_utils import Cards, NeuralNet
import numpy as np
import cv2
from random import randint, choice

x = 30
y = 70

if __name__ == '__main__':
    nn = NeuralNet(x, y)
    clf = nn.train()
    cam = cv2.VideoCapture(0)

    while cam.isOpened():
        ret, img = cam.read()
        cards = Cards(img)
        corners, contours, hierarchy = cards.getFeatures((9, 9))
        suits = cards.extractSuits(corners)
        for suit in suits:
            suit = cv2.resize(suit, (x, y))
            prediction = clf.predict(suit.reshape(1, x*y))
            print(prediction)
            cv2.imshow('suit', suit)
        prediction = []
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        cv2.imshow('cam', img)

        if cv2.waitKey(1) & 0xFF == ord('o'):
            option = str(input('What would you like to do? '))
            if option is 't':
                #retrain
                nn.retrain(suit.reshape(1, x*y))
                pass
            elif option is 'x':
                # Quit program
                break

    if str(input('Would you like to save the coefficients? yes/no ')) == 'yes':
            np.save('coefs_', clf.coefs_)
            print('Coeffients save in coefs_.npy')

    cam.release()
    cv2.destroyAllWindows()
