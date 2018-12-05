from card_utils import Cards, NeuralNet
import numpy as np
import cv2
from random import randint, choice



if __name__ == '__main__':
    # Image size. Do not change! Unless you deleted both coefs_.npy and intercepts_.npy file.
    x = 30
    y = 70

    nn = NeuralNet((100, 100, 100), 0.00001)
    clf = nn.train()
    # Set cv2.VideoCapture parameter to 1 if using USB webcam. Recommended!!
    cam = cv2.VideoCapture(1)

    while cam.isOpened():
        ret, img = cam.read()
        cards = Cards(img)
        corners, contours, hierarchy = cards.getFeatures((9, 9))
        # Extracting a preprocced image containing both suits and rank of the card.
        suits = cards.extractSuits(corners)
        for i in range(len(suits)):
            suit = cv2.resize(suits[i], (x, y))
            prediction = clf.predict(suit.reshape(1, x*y))
            cv2.putText(img, prediction[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
            print(prediction)
            cv2.imshow('suit', suit)
        prediction = []
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        cv2.imshow('cam', img)
        
        # Pressing 'o' (options) in the 'Camera Image' window a UI option menu will show at the terminal.
        if cv2.waitKey(1) & 0xFF == ord('o'):
            option = str(input('What would you like to do? '))
            # Train option, unfortunatily once trained it looks at that suit & rank only. Not working!!!
            if option == 'train':
                # retrain dataset with specific label! Runs but classifications is no longer accurate
                print(suit.shape)
                cv2.imshow('Unknown Card!', suit)
                target = str(input('What card is this?'))
                nn.retrain(suit.reshape(1, x*y), np.array([target]))
            elif option == 'exit':
                # Quit program
                break
    
    # Saving Coefficients and Intercepts values UI. Becarefull not to overfit with this.
    if str(input('Would you like to save the coefficients & intercepts? yes/no ')) == 'yes':
            np.save('../data/coefs_', clf.coefs_)
            np.save('../data/intercepts_', clf.intercepts_)
            print('Coeffients save in ../data/coefs_.npy')
            print('Intercepts save in ../data/intercepts_.npy')

    cam.release()
    cv2.destroyAllWindows()
