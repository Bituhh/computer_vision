from sklearn import svm
import numpy as np
import cv2
from random import randint, choice

suits = ['h', 's', 'd', 'c']


# creating a list from the dataset and label.
imgs = []
labels = []
for suit in range(4):
    for number in range(1, 14):
        for card_index in range(50):
            label = str(number) + suits[suit] 
            img = cv2.imread('../images/' + str(card_index) + '-' + label + '.png')
            labels.append(label)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


reshaped_imgs = np.array(imgs).reshape(len(imgs), -1)
print(np.array(imgs).shape)
print(reshaped_imgs.shape)
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(reshaped_imgs, labels)
correct = 0
# Test predition
for x in range(5000):
    i = randint(0, len(imgs)-1)    
    val = clf.predict(imgs[i].reshape(1, 2100))
    if val[0] == labels[i]:
        correct = correct + 1
    
    print(str((correct // (x+1))*100) + '%' )
