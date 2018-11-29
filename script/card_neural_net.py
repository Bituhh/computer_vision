from card_utils import Cards
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import cv2
from random import randint, choice

x = 30
y = 70

scaler = StandardScaler()

def train():
    print('Building dataset array...')
    suits = ['h', 's', 'd', 'c']
    imgs = []
    labels = []
    


    # creating a training list from the dataset and label.
    for suit in suits:
        for rank in range(1, 14):
            for card_index in range(50):
                label = str(rank) + suit
                img = cv2.imread('../images/' + str(card_index) + '-' + label + '.png', 0)
                img = cv2.resize(img, (x, y))
                #thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2, 2)
                #cv2.imshow('img', img)
                #cv2.waitKey(0)
                labels.append(label)

                imgs.append(img)#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    

    
    # reshape the image into a vector array. shape = (2600 ,2100)
    #print('Reshaping and applying PCA to image array! ...')
    reshaped_imgs = np.array(imgs).reshape(len(imgs), -1)
    scaler.fit(reshaped_imgs)
    imgs = scaler.transform(reshaped_imgs)
    #mean, eigenVectors = cv2.PCACompute(reshaped_imgs, mean=None, maxComponents=10)
    #reshaped_imgs = eigenVectors.reshape(-1, len(eigenVectors))
    #kernel=1.0 * gaussian_process.kernels.RBF(length_scale=10.0)
    #print(eigenVectors.shape)
    print(reshaped_imgs.shape)
    print('Dataset array complete!') 
    
    print('Training MLPClassifier...')
    # initialising and training NN
    clf = MLPClassifier(hidden_layer_sizes=(1000, 1000, 1000, 1000), random_state=1, shuffle=True,learning_rate_init=0.001)
    clf.fit(imgs, labels)
    #print(clf.get_params)
    print('Training Complete!')
    return clf, clf.get_params(True), imgs, labels

if __name__ == '__main__':
    
    
    clf, params, imgs, labels = train()
    cam = cv2.VideoCapture(1)

    while cam.isOpened():
        ret, img = cam.read()
        cards = Cards(img)
        corners, contours, hierarchy = cards.getFeatures((5, 5), 90000)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        suits = cards.extractSuits(corners)
        for suit in suits:
            suit = cv2.resize(suit, (x, y))
            prediction = clf.predict(scaler.transform(suit.reshape(1, x*y)))
            #print(hierarchy)
            print(prediction)
            cv2.imshow('suit', suit)
        prediction = []
        cv2.imshow('cam', img)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break


    correct = 0
    # Test NN
    for j in range(5000):
        break
        i = randint(0, len(imgs)-1)    
        val = clf.predict(imgs[i].reshape(1, x*y)) # reshape test imgs to fit sklearn classifier
        if val[0] == labels[i]:
            correct = correct + 1
        
        print(str((correct // (j+1))*100) + '%' ) # print accurate
    print(clf.coefs_)
    np.save('coefs_', clf.coefs_)

    print(np.load('coefs_.npy'))

    cam.release()
    cv2.destroyAllWindows()

