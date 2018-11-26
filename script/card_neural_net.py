from card_utils import Cards
from sklearn import svm
import numpy as np
import cv2
from random import randint, choice

def train():
    suits = ['h', 's', 'd', 'c']
    imgs = []
    labels = []
    
    # creating a training list from the dataset and label.
    for suit in range(4):
        for number in range(1, 14):
            for card_index in range(50):
                label = str(number) + suits[suit] 
                img = cv2.imread('../images/' + str(card_index) + '-' + label + '.png', 0)
                thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2, 2)
                #cv2.imshow('img', img)
                #cv2.waitKey(0)
                labels.append(label)

                imgs.append(thresh)#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


    # reshape the image into a vector array. shape = (2600 ,2100)
    reshaped_imgs = np.array(imgs).reshape(len(imgs), -1)
    
    # initialising and training NN
    clf = svm.SVC(gamma=0.001, C=1000)
    clf.fit(reshaped_imgs, labels)
    return clf 

if __name__ == '__main__':
    
    clf = train()
    kernel = tuple([5, 5])
    cam = cv2.VideoCapture(0)

    while cam.isOpened():
        ret, img = cam.read()
        cards = Cards(img)
        corners, contours, hierarchy = cards.getFeatures(kernel)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        suits = cards.extractSuits(corners)
        for suit in suits:
            prediction = clf.predict(suit.reshape(1, 2100))
            print(prediction)
            cv2.imshow('suit', suit)
        cv2.imshow('cam', img)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break


   
    # Test NN
    for x in range(5000):
        i = randint(0, len(imgs)-1)    
        val = clf.predict(imgs[i].reshape(1, 2100)) # reshape test imgs to fit sklearn classifier
        if val[0] == labels[i]:
            correct = correct + 1
        
        print(str((correct // (x+1))*100) + '%' ) # print accurate

        print(clf.coef_)

    cam.release()
    cv2.destroyAllWindows()

