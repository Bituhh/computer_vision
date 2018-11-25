from world_features_multiclassifier import *
#from operator import truediv
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

classifiers = [KNeighborsClassifier(3),
DecisionTreeClassifier(),
GaussianNB(),
LinearDiscriminantAnalysis(),
QuadraticDiscriminantAnalysis(),
MLPClassifier()]

classifier = classifiers[0]
# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
#classifier = MLPClassifier(activation='logistic', solver='sgd', alpha=1e-5, hidden_layer_sizes=(10,))
print(type(classifier))


dataset = 20
in_batch = 10
sequence = list(range(20))
random.shuffle(sequence)
#sequence = random.shuffle(range(20))

logfile = open('log3.txt', 'w')
# initialization
folder = './images2/'

#world_obj = ['box','spoon','knife','circle']
world_obj = ['A','B','C','D','E','F']
shape_names = ['cylinder','cube','triangle','short_rect','long_rect','bridge']

# colours = [] # for black and white
colours = ['wood','red','yellow','blue','dark_green','light_green']

hsv_min = [[0.0, 0.0, 231.0],[0.0, 0.0, 180.0],[29.7, 0.0, 218.0],[98.5, 0.0, 180.0],[60.0, 0.0, 90.0],[44.3, 0.0, 220.0]]
hsv_max = [[55.0, 255.0, 255.0],[29.7, 255.0, 255.0],[33.2, 255.0, 255.0],[101.1, 255.0, 255.0],[80.0, 255.0, 255.0],[54.2, 255.0, 255.0]]

features_dict = ['axes_ratio','concavity_ratio','convexity_ratio','area_ratio','vertex_approx','length','perimeter_ratio']
use_features = [0, 1, 2, 3, 4, 5, 6]#
features = [features_dict[ft] for ft in use_features]
decision_threshold = 0.00001 #0.00001

n = len(world_obj)
examples = [sequence[sq] for sq in range(in_batch)] # [10, 12, 11, 12]
current_world = world(folder,world_obj,examples,features,logfile,colours,hsv_min,hsv_max,classifier)

# test phase
examples = np.tile(dataset, n) # [10, 12, 11, 12]
update_world = False
correct = np.tile(0, n)
thr_correct = np.tile(0, n)
uncertain = np.tile(0, n)

for s in range(in_batch, dataset):
    for obj in range(n):
        filename = world_obj[obj] + str(sequence[s])
        probs = current_world.update(folder,filename,features,update_world,logfile,colours,obj)  # obj = -1 means colour unknown
        correct[obj] = correct[obj] + (np.argmax(probs)==obj)
        #print colours[obj]
        #print world_obj
        #print probs
        #print
        
test_ex = [dataset-in_batch for x in examples]
correct_rel = correct/test_ex # correct classifications
thr_correct_rel = thr_correct/(test_ex - uncertain) # correct classifications for over threshold tests
thr_correct_tot = thr_correct/test_ex # correct classifications over threshold
print(["%0.2f" % p for p in correct_rel], "%0.2f" % np.mean(correct_rel))
#print ["%0.2f" % p for p in thr_correct_rel], "%0.2f" % np.mean(thr_correct_rel)
#print ["%0.2f" % p for p in thr_correct_tot], "%0.2f" % np.mean(thr_correct_tot)
logfile.close()

#(pca, X) = current_world.run_pca()
#print(pca.explained_variance_ratio_)
#print X
