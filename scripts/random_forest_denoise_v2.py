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
images_train = np.array(pickle.load(open(parent_dir+"/data/train_cleaned_examples.p", "rb")))

#print("Images %s \n Shape %s \n"%(images_train,images_train.shape))


#Normalize the pictures: 41x41
#Add nb_row rows of 0 below and nb_colums colums of 0 at the right to have X of size 41x41

n = len(images_train)
n2 = images_train[0].shape
size = 41
X= np.zeros([n,size,size])

for i in range(n):
    x_size = images_train[i].shape
    nb_row = size-x_size[0]
    nb_columns = size-x_size[1]
    x = np.pad(images_train[i], ((0,nb_row),(0,nb_columns)), mode='constant', constant_values=0)
    X[i] = np.pad(images_train[i], ((0,nb_row),(0,nb_columns)), mode='constant', constant_values=0)

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
images_train = images_train.reshape(8000,1681)
images_validation = images_validation.reshape(2000,1681)

train_labels = y[0:8000]
validation_labels = y[8000:]

clf = RandomForestClassifier(random_state=0, n_estimators=500, max_depth=15, min_samples_split = 2)
    #    'min_samples_split': [3, 5, 10],
    #       'n_estimators' : [100, 300],
    #    'max_depth': [3, 5, 15, 25],
#  'max_features': [3, 5, 10, 20]
#param_grid = {
#    'bootstrap': [True],
#    'max_depth': [80, 90, 100, 110],
#    'max_features': [2, 3],
#    'min_samples_leaf': [3, 4, 5],
#    'min_samples_split': [8, 10, 12],
#    'n_estimators': [100, 200, 300, 1000]
#}

#GridSearch
#rfc = RandomForestClassifier(random_state=42)
#best params:  {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'auto', 'n_estimators': 500}
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

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
