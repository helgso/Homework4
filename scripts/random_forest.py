import numpy as np # linear algebra
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import csv
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__)))

#Load images with numpy (training set)
images_train = np.load(parent_dir+'/data/train_images.npy', encoding='latin1')
#print("Shape: %s, Images_train: %s"%(images_train.shape,images_train))

#Load labels
train_labels = np.genfromtxt(parent_dir+'/data/train_labels.csv', names=True, delimiter=',', dtype=[('Id', 'i8'), ('Category', 'S5')])
#print("Shape: %s, train_labels: %s"%(train_labels.shape,train_labels))

#Load images with numpy (test set)
images_test = np.load(parent_dir+'/data/test_images.npy', encoding='latin1')

#Below we'll just convert data to get X: [[vect1], [vect2], ..] and y = [labxˆ1,labxˆ2,...,labxˆn]

X_ = images_train[:,-1]

n = len(X_)
n2 = len(X_[0])

y = [x[1] for x in train_labels]

X= np.zeros([n,n2])

for i in range(n):
    X[i] = X_[i].tolist()

#print(X)

#Try different max_depth (tree max depth) to improve results
clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             random_state=0)
clf.fit(X, y)
acc = clf.score(X, y)
print("Accuracy on the training set: %s \n"%acc)

print("clf.feature_importances: %s \n "%clf.feature_importances_)

Xtest_ = images_train[:,-1]
nt = len(Xtest_)
nt2 = len(Xtest_[0])
Xtest= np.zeros([nt,nt2])
for i in range(nt):
    Xtest[i] = Xtest_[i].tolist()

results = clf.predict(Xtest)

print("Results (predictions on the test set): %s \n"%results)

np.savetxt(parent_dir+'/results/test_random_forest.csv', [p for p in results], delimiter=' ', fmt='%s')

print("Predictions as been saved as a csv file")
