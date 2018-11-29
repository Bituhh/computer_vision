from card_utils import Cards
from sklearn.neural_network import MLPClassifier
import numpy as np
import cv2
from random import randint, choice

x = 30
y = 70

def build_dataset(x, y):
    print('Building dataset array...')
    
    # Creating a dataset and label lists.
    
    suits = ['h', 's', 'd', 'c']
    imgs = []
    labels = []
    
    for suit in suits:
        for rank in range(1, 14):
            for card_index in range(250):
                label = str(rank) + suit
                img = cv2.imread('../images/' + str(card_index) + '-' + label + '.png', 0)
                img = cv2.resize(img, (x, y))
                labels.append(label)
                imgs.append(img)
 
    # Reshape the image into a vector array.
    
    reshaped_imgs = np.array(imgs).reshape(len(imgs), -1)
    print('Dataset array complete!')
    return reshaped_imgs, labels
 
def train():
    print('Building MLPClassifier ...')
    
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1, batch_size=500, learning_rate_init=0.01, verbose=True, n_iter_no_change=100, shuffle=True)
    
    try:
        print('Attempting to load pre-trained coeffiencts...')
        coefs = np.load('coefs_.npy')
        
        if str(input('Coefficients found! Would you like to load it? yes/no')) == 'yes':
            clf.coefs_ = coefs
            print('Coefficients loaded successfully!')
        else:
            imgs, labels = build_dataset(x, y)
            print('Training MLPClassifier...')
            clf.fit(imgs, labels)
            print('Training Complete!')
    
    except:
        print('No pre-trained biases and coeffiencients found!')
        print('Training MLPClassifier...')
        clf.fit(reshaped_imgs, labels)
        print('Training Complete!')
    
    print('MLPClassifier successfully built!')
    return clf, clf.get_params(True)

if __name__ == '__main__':
    clf, params = train()
    cam = cv2.VideoCapture(0)

    while cam.isOpened():
        ret, img = cam.read()
        cards = Cards(img)
        corners, contours, hierarchy = cards.getFeatures((9, 9))
        suits = cards.extractSuits(corners)
        for suit in suits:
            suit = cv2.resize(suit, (x, y))
            prediction = clf.predict(suit.reshape(1, x*y))
            #print(hierarchy)
            print(prediction)
            cv2.imshow('suit', suit)
        prediction = []
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        cv2.imshow('cam', img)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break


    if str(input('Would you like to save the coefficients? yes/no ')) == 'yes':
            np.save('coefs_', clf.coefs_)
            print('Coeffients save in coefs_.npy')
    
    np.load('coefs_.npy')

    cam.release()
    cv2.destroyAllWindows()

