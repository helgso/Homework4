import numpy as np # linear algebra
import matplotlib.pyplot as plt
import os
import csv
from os.path import dirname, abspath
import pickle
from sklearn.model_selection import cross_val_score
from math import *
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
import keras


parent_dir = dirname(dirname(abspath(__file__)))

#Load pickle denoised training set
images_train = np.array(pickle.load(open(parent_dir+"/data/train_cleaned_examples.p", "rb")))

#print("Images %s \n Shape %s \n"%(images_train,images_train.shape))


#Normalize the pictures: 46x46 = add 0 to fill

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

#Reshape to use the convolutional network
X = X.reshape(10000,size,size,1)

#Load labels for training
train_labels = np.genfromtxt(parent_dir+'/data/train_labels.csv',delimiter=',',dtype=None, names=True)
Y_ = [x[1] for x in train_labels]
one_hot = pd.get_dummies(Y_, sparse = True)
one_hot_labels = np.asarray(one_hot)

#Transform predictions in onehot
Y = np.zeros([n,one_hot_labels.shape[1]])

for i in range(n):
    label = one_hot_labels[i]
    Y[i] = label

num_class = Y.shape[1]

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=1)

# Create the base pre-trained model
# Can't download weights in the kernel
#weights='imagenet' -> need to resize the data
base_model = VGG19(weights=None, include_top=False, input_shape=(size,size,1))

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()

model.fit(X_train, Y_train, epochs=30, batch_size=64, validation_data=(X_valid, Y_valid), verbose=1)

