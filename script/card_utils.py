import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier

class Cards:
    def __init__(self, img):
        self.img = img

    def getFeatures(self, kernel_shape, min_area = 2000, max_area = 120000):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_shape)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray,11,17,17)
        blur = cv2.GaussianBlur(gray, kernel_shape, 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2, 2) 
        erode = cv2.erode(thresh, kernel, iterations = 1)
        canny = cv2.Canny(blur, blur.mean()*0.01, blur.mean()*1.5)
        img, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cards_contours = []
        cards_corners = []
        cards_hierarchy = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            perimeter = 0.1*cv2.arcLength(contours[i],True)
            corner = cv2.approxPolyDP(contours[i],perimeter,True)
            if (area > min_area) and (area < max_area) and (len(corner) == 4) and (hierarchy[0, i, 3] == -1):    
                cards_contours.append(contours[i])
                cards_corners.append(corner)
                cards_hierarchy.append(hierarchy[0])
        return cards_corners, cards_contours, cards_hierarchy

    
    def extractCard(self, corner):
        a_dis = abs(corner[0, 0, 0] - corner[1, 0, 0]) + abs(corner[0 , 0, 1] - corner[1, 0, 1])
        b_dis = abs(corner[1, 0, 0] - corner[2, 0, 0]) + abs(corner[1 , 0, 1] - corner[2, 0, 1])
        if a_dis > b_dis:
            pts1 = np.float32([corner[0, 0], corner[3, 0], corner[1, 0], corner[2, 0]])
        else:
            pts1 = np.float32([corner[1, 0], corner[0, 0], corner[2, 0], corner[3, 0]])
        pts2 = np.float32([[0,0],[200, 0], [0, 300], [200, 300]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(self.img, matrix, (200, 300))

    def extractSuits(self, corners):
        suits = []
        if len(corners) is not 0:
            for corner in corners:
                card = self.extractCard(corner)
                gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray,11,17,17)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2, 2)

                suits.append(thresh[7:77, 5:35])
        return suits

class NeuralNet:
    def __init__(self, img_size_x, img_size_y):
        self.x = img_size_x
        self.y = img_size_y

    def build_dataset(self):
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
                    img = cv2.resize(img, (self.x, self.y))
                    labels.append(label)
                    imgs.append(img)
     
        # Reshape the image into a vector array.
        
        reshaped_imgs = np.array(imgs).reshape(len(imgs), -1)
        print('Dataset array complete!')
        return reshaped_imgs, labels
     
    def train(self):
        print('Building MLPClassifier ...')
        train = False
        clf = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1, batch_size=500, learning_rate_init=0.01, verbose=True, n_iter_no_change=50, shuffle=True)
        
        try:
            print('Attempting to load pre-trained coeffiencts...')
            coefs = np.load('coefs_.npy')
            if str(input('Coefficients found! Would you like to load it? yes/no')) == 'yes':
                clf.coefs_ = coefs
                train = False
                print('Coefficients loaded successfully!')
            else:
                train = True
        
        except:
            print('No pre-trained biases and coeffiencients found!')
            train = True
        
        if train:
            imgs, labels = self.build_dataset()
            print('Training MLPClassifier...')
            clf.fit(imgs, labels)
            print('Training Complete!')
        
        print('MLPClassifier successfully built!')
        return clf

