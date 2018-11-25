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

parent_dir = dirname(dirname(abspath(__file__)))

#Load pickle denoised training set
images_train = np.array(pickle.load(open(parent_dir+"/data/train_cleaned_examples.p", "rb")))

#Load pickle denoised testing set
images_test = np.array(pickle.load(open(parent_dir+"/data/test_cleaned_examples.p", "rb")))

#Normalize the pictures: 46x46
#Add nb_row rows of 0 below and nb_colums colums of 0 at the right to have X of size 46x46

n = len(images_train)
n2 = images_train[0].shape
size = 46
X= np.zeros([n,size,size])

for i in range(n):
    x_size = images_train[i].shape
    nb_row = size-x_size[0]
    nb_columns = size-x_size[1]
    x = np.pad(images_train[i], ((0,nb_row),(0,nb_columns)), mode='constant', constant_values=0)
    X[i] = np.pad(images_train[i], ((0,nb_row),(0,nb_columns)), mode='constant', constant_values=0)

#print(X)
#Normalise pictures of test set too
n_ = len(images_test)
n2_ = images_test[0].shape
size_ = 46
X_= np.zeros([n_,size_,size_])

for i in range(n_):
    x_size = images_test[i].shape
    nb_row = size_-x_size[0]
    nb_columns = size_-x_size[1]
    x = np.pad(images_test[i], ((0,nb_row),(0,nb_columns)), mode='constant', constant_values=0)
    X_[i] = np.pad(images_test[i], ((0,nb_row),(0,nb_columns)), mode='constant', constant_values=0)

#print(X_)

images_train = X
images_test = X_

#Load labels for training
train_labels = np.genfromtxt(parent_dir+'/data/train_labels.csv',delimiter=',',dtype=None, names=True)
print(train_labels)
y = [x[1] for x in train_labels]

#We want matrixes of vectors
images_train = images_train.reshape(10000,2116)
images_test = images_test.reshape(10000,2116)

train_labels = y

clf = RandomForestClassifier(random_state=0, n_estimators=500, max_depth=15)

clf.fit(images_train, train_labels)

#print("best params: ", clf.best_params_)

acc = clf.score(images_train, train_labels)
print("Accuracy on the training set: %s"%acc)

#Uncomment to use cross validation
#cv1 = cross_val_score(clf, images_train, train_labels, scoring='recall_macro',cv=5)
#print("Accuracy on the training set: %s \n %s \n"%(acc,cv1))

print("clf.feature_importances: %s \n "%clf.feature_importances_)

results = clf.predict(images_test)

print("Results (predictions on the test set): %s \n"%results)

np.savetxt(parent_dir+'/results/test_random_forest_v4_denoise.csv', [p for p in results], delimiter=' ', fmt='%s')

print("Predictions as been saved as a csv file")
