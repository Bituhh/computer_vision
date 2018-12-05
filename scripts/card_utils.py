import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

class Cards:
    def __init__(self, img):
        self.img = img

    def getFeatures(self, kernel_shape, min_area = 2000, max_area = 120000):
        # Preprocessing image to get features.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_shape)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray,11,17,17) # Filtering noise
        blur = cv2.GaussianBlur(gray, kernel_shape, 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2, 2) 
        erode = cv2.erode(thresh, kernel, iterations = 1)
        canny = cv2.Canny(blur, blur.mean()*0.01, blur.mean()*1.5)
        img, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialising return lists
        cards_contours = []
        cards_corners = []
        cards_hierarchy = []
        
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            perimeter = 0.1*cv2.arcLength(contours[i],True)
            corner = cv2.approxPolyDP(contours[i],perimeter,True)
            
            # Contour and corners will only be appended to the list, only
            # if the area is within range (max_area, min_area), contains 4
            # corners and have no parents (hierarchy).
            if (area > min_area) and (area < max_area) and (len(corner) == 4) and (hierarchy[0, i, 3] == -1):    
                cards_contours.append(contours[i])
                cards_corners.append(corner)
                cards_hierarchy.append(hierarchy[0])
        
        return cards_corners, cards_contours, cards_hierarchy

    
    def extractCard(self, corner):
        # Determining the size of height and width of the card by adding its x and y positions
        a_dis = abs(corner[0, 0, 0] - corner[1, 0, 0]) + abs(corner[0 , 0, 1] - corner[1, 0, 1])
        b_dis = abs(corner[1, 0, 0] - corner[2, 0, 0]) + abs(corner[1 , 0, 1] - corner[2, 0, 1])
        # Flipping corners to fix contours having an external and internal side, causing the warp to flip.
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
                # Preprocessing extracted cards.
                gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray,11,17,17)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2, 2)

                # Appending cropped image of suit and rank to suits array.
                suits.append(thresh[7:77, 5:35])
        return suits

class NeuralNet(MLPClassifier):
    def __init__(self, hidden_layer, rate, n_sample=100):
        # Inicialising MLPClassifier as a part of the Neural Net class. This
        # was done to allow overwriting of the _init_coef function from
        # MLPClassifier.
        super().__init__(hidden_layer_sizes=hidden_layer, random_state=1, batch_size=100, learning_rate_init=rate, verbose=True, n_iter_no_change=50, shuffle=True, max_iter=200, learning_rate='adaptive')

        self.x = 30
        self.y = 70
        self.counter = 0
        self.n_sample = n_sample

    # Overwriting _init_coef so that a pre-trained neural net can be utilised. Working!
    def _init_coef(self, fan_in, fan_out):
        if self.activation == 'logistic':
            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation in ('identity', 'tanh', 'relu'):
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        else:
            raise ValueError("Unknown activation function %s" % self.activation)
        
        try:
            print('Attemping to load pre-trained coefficients and intercept...')
            coef = np.load('../data/coefs_.npy')
            intercept = np.load('../data/intercepts_.npy')
            coef_init = coef[self.counter]
            intercept_init = intercept[self.counter]
            self.counter = self.counter + 1
            print('Coefficients and intercept loaded successfully!')
        except:
            print('pre-trained coefficients and intercept not fould! Inicialising ramdonly...')
            coef_init = self._random_state.uniform(-init_bound, init_bound, (fan_in, fan_out))
            intercept_init = self._random_state.uniform(-init_bound, init_bound, fan_out)
        
        return coef_init, intercept_init


    def build_dataset(self):
        print('Building dataset array...')
        # Creating the dataset and label lists.
        suits = ['h', 's', 'd', 'c']
        imgs = []
        labels = []
        
        for suit in suits:
            for rank in range(1, 14):
                for card_index in range(self.n_sample):
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

#       Attempting to create a pre_trained initialiser. Decided to overwrite  _init_coef!!!
#        try:
#            print('Attempting to load pre-trained coeffiencts...')
#            if str(input('Coefficients found! Would you like to load it? yes/no ')) == 'yes':
#                self.clf.coefs_ = coefs
#                self.clf.n_outputs_ = 52
#                self.clf.n_layers_ = 4
#                self.clf.intercepts_ = intercepts
#                self.clf.out_activation_ = 'softmax'
#                self.clf._label_binarizer = LabelBinarizer()
#                train = False
#                print('Coefficients loaded successfully!')
#            else:
#                train = True
#        
#        except:
#            print('No pre-trained biases and coeffiencients found!')
#            train = True
        
        imgs, labels = self.build_dataset()
        print('Training MLPClassifier...')
        self.fit(imgs, labels)
        print('Training Complete!')
        #print(self.)
        print('MLPClassifier successfully built!') 
        return self

    # CAUTION! Retraining will cause the neural net to overfit to the specified target.
    def retrain(self, img, target):
        self.batch_size = 1
        self.max_iter = 1
        self.fit(img, target)

