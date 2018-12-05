import cv2
import numpy as np
import random


# This file performed all transformations to the pre-existing dataset, this was done to consider the lack of movement and rotation from the dataset!


def transform_dataset():
    print('Building dataset array...')
    # Creating the dataset and label lists.
    suits = ['h', 's', 'd', 'c']
    imgs = []
    labels = []
    
    for suit in suits:
        for rank in range(1, 14):
            for card_index in range(250):
                label = str(rank) + suit
                img = cv2.imread('../images/' + str(card_index) + '-' + label + '.png', 0)
                img = cv2.resize(img, (30, 70))
                translate = np.float32([[1, 0, random.randint(-8, 8)], [0, 1, random.randint(-8, 8)]])
                translated_img = cv2.warpAffine(img, translate, (30, 70))
                rotate = cv2.getRotationMatrix2D((15, 35), random.randint(-8, 8), 1)
                rotated_img = cv2.warpAffine(translated_img, rotate, (30, 70))
                cv2.imwrite('../images/' + str(card_index) + '-' + label + '.png', rotated_img)
                #cv2.imshow('trans', translated_img)
                #cv2.imshow('rotated', rotated_img)
                #cv2.waitKey(0)
    # Reshape the image into a vector array.

transform_dataset()
cv2.destroyAllWindows()
