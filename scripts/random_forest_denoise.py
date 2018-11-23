import numpy as np # linear algebra
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import csv
from os.path import dirname, abspath
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

parent_dir = dirname(dirname(abspath(__file__)))

#Load pickle denoised training set
images_train = np.array(pickle.load(open(parent_dir+"/data/cross-validation/D_noised_Validation_train_set.p", "rb")))

#Load pickle labels for training
train_labels = np.array(pickle.load(open(parent_dir+"/data/cross-validation/Validation_train_labels_categories.p", "rb")))

#Load denoised test set
images_test = np.array(pickle.load(open(parent_dir+"/data/cross-validation/D_noised_Validation_test_set.p", "rb")))

#Load labels of the test set to compare with the prediction
test_labels = np.array(pickle.load(open(parent_dir+"/data/cross-validation/Validation_test_labels_categories.p", "rb")))

#We want 100x100 pictures instead of 100x105
images_train = images_train[:,:,3:-2].reshape(8000,10000)
images_test = images_test[:,:,3:-2].reshape(2000,10000)

#clf = RandomForestClassifier(random_state=42, criterion='entropy', max_features='auto', n_estimators=500, max_depth=8)

#GridSearch
rfc = RandomForestClassifier(random_state=42)
#best params:  {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'auto', 'n_estimators': 500}
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3, 'auto'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
clf.fit(images_train, train_labels)

print("best params: ", clf.best_params_)

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
