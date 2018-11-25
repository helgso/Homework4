import numpy as np # linear algebra
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import csv
from os.path import dirname, abspath
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from math import *

parent_dir = dirname(dirname(abspath(__file__)))

#Load pickle denoised training set
images_train = np.array(pickle.load(open(parent_dir+"/data/train_cleaned_examples.p", "rb")))

#print("Images %s \n Shape %s \n"%(images_train,images_train.shape))


#Normalize the pictures: 41x41
#Add nb_row rows of 0 below and nb_colums colums of 0 at the right to have X of size 41x41

n = len(images_train)
n2 = images_train[0].shape
size = 46
X= np.zeros([n,size,size])

for i in range(n):
    x_size = images_train[i].shape
    nb_row = size-x_size[0]
    nb_columns = size-x_size[1]
    zero_up = ceil(nb_row/2)
    zero_down = floor(nb_row/2)
    zero_left = ceil(nb_columns/2)
    zero_right = floor(nb_columns/2)
    x = np.pad(images_train[i], ((zero_up,zero_down),(zero_left,zero_right)), mode='constant', constant_values=0)
    X[i] = x

#print(X)

#Test our model with the validation set
images_train = X[0:8000,:]
images_validation = X[8000:,:]

#Load labels for training
train_labels = np.genfromtxt(parent_dir+'/data/train_labels.csv', names=True, delimiter=',', dtype=[('Id', 'i8'), ('Category', 'S5')])
y = [x[1] for x in train_labels]

#Load denoised test set
images_test = np.array(pickle.load(open(parent_dir+"/data/test_cleaned_examples.p", "rb")))

#We want matrixes of vectors
images_train = images_train.reshape(8000,2116)
images_validation = images_validation.reshape(2000,2116)

train_labels = y[0:8000]
validation_labels = y[8000:]


clf = RandomForestClassifier(random_state=1, n_estimators=500, max_depth=30)
#bclf = AdaBoostClassifier(base_estimator=clf,n_estimators=clf.n_estimators)

#clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
clf.fit(images_train, train_labels)

#print("best params: ", clf.best_params_)

acc = clf.score(images_train, train_labels)
print("Accuracy on the training set: %s"%acc)

#Uncomment to use cross validation
#cv1 = cross_val_score(clf, images_train, train_labels, scoring='recall_macro',cv=5)
#print("Accuracy on the training set: %s \n %s \n"%(acc,cv1))

acc2 = clf.score(images_validation, validation_labels)
print("Accuracy on the test set: %s"%acc2)

#Uncomment to use cross validation
#cv2 = cross_val_score(clf, images_test, test_labels, scoring='recall_macro',cv=5)
#print("Accuracy on the training set: %s \n %s \n"%(acc2,cv2))

print("clf.feature_importances: %s \n "%clf.feature_importances_)

#results = clf.predict(images_test)

#print("Results (predictions on the test set): %s \n"%results)

#np.savetxt(parent_dir+'/results/test_random_forest_v2_denoise.csv', [p for p in results], delimiter=' ', fmt='%s')

#print("Predictions as been saved as a csv file")
