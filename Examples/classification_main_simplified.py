# classifying simple objects based on a selected set of features

from world_features import *
import random

# initialization
folder = './images2/'
logfile = open('log.txt', 'w')
world_obj = ['A','B','C','D','E','F']
n = len(world_obj)
shape_names = ['cylinder','cube','triangle','short_rect','long_rect','bridge']
colours = ['wood','red','yellow','blue','dark_green','light_green']

# colour ranges used in colour segmentation
hsv_min = [[0.0, 0.0, 231.0],[0.0, 0.0, 180.0],[29.7, 0.0, 218.0],[98.5, 0.0, 180.0],[60.0, 0.0, 90.0],[44.3, 0.0, 220.0]]
hsv_max = [[55.0, 255.0, 255.0],[29.7, 255.0, 255.0],[33.2, 255.0, 255.0],[101.1, 255.0, 255.0],[80.0, 255.0, 255.0],[54.2, 255.0, 255.0]]

# features used in object classification
features_dict = ['axes_ratio','concavity_ratio','convexity_ratio','area_ratio','vertex_approx','length','perimeter_ratio']
use_features = [0, 1, 2, 3, 4, 5, 6] # 0..6
features = [features_dict[ft] for ft in use_features]

# create world experience with 10 examples for each object
in_batch = 10
sequence = list(range(in_batch))
random.shuffle(sequence)
examples = [sequence[sq] for sq in range(in_batch)]
current_world = world(folder,world_obj,examples,features,logfile,colours,hsv_min,hsv_max)

# test phase
obj = 1 # 0..5
s = 11 # 1..10
update_world = False
filename = world_obj[obj] + str(s)
probs = current_world.update(folder,filename,features,update_world,logfile,colours,obj)  # obj = -1 means colour unknown
print('True: ', shape_names[obj])
print('Guess: ', shape_names[np.argmax(probs)])
print(["%0.6f" % p for p in probs])
logfile.close()
