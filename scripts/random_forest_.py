import numpy as np # linear algebra
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import csv
from os.path import dirname, abspath
import pickle

parent_dir = dirname(dirname(abspath(__file__)))

#Load pickle denoised training set
images_train = np.array(pickle.load(open(parent_dir+"/data/cross-validation/Validation_train_set.p", "rb")))

#Load pickle labels for training
train_labels = np.array(pickle.load(open(parent_dir+"/data/cross-validation/Validation_train_labels_categories.p", "rb")))

#Load denoised test set
images_test = np.array(pickle.load(open(parent_dir+"/data/cross-validation/Validation_test_set.p", "rb")))

#Load labels of the test set to compare with the prediction
test_labels = np.array(pickle.load(open(parent_dir+"/data/cross-validation/Validation_test_labels_categories.p", "rb")))

images_train = images_train.reshape(8000,10000)
images_test = images_test.reshape(2000,10000)

#Try different max_depth (tree max depth) to improve results
clf = RandomForestClassifier(n_estimators=20, max_depth=50,max_features=1000,
                             random_state=0)
clf.fit(images_train, train_labels)
acc = clf.score(images_train, train_labels)
print("Accuracy on the training set: %s"%acc)

#Uncomment to use cross validation
#cv1 = cross_val_score(clf, images_train, train_labels, scoring='recall_macro',cv=5)
#print("Accuracy on the training set: %s \n %s \n"%(acc,cv1))

acc2 = clf.score(images_test, test_labels)
print("Accuracy on the test set: %s"%acc2)

#Uncomment to use cross validation
#cv2 = cross_val_score(clf, images_test, test_labels, scoring='recall_macro',cv=5)
#print("Accuracy on the training set: %s \n %s \n"%(acc2,cv2))

print("clf.feature_importances: %s \n "%clf.feature_importances_)

results = clf.predict(images_test)

print("Results (predictions on the test set): %s \n"%results)

np.savetxt(parent_dir+'/results/test_random_forest_v2_denoise.csv', [p for p in results], delimiter=' ', fmt='%s')

print("Predictions as been saved as a csv file")
