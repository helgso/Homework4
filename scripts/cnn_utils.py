import numpy as np

# Requires python 3, and no higher than 3.6.x
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import random
import pickle


# Where we keep track of which label is which class number (integer)
LABEL_TO_INTEGER = {}
LABELS = []


def load_training_dataset(
    img_size,
    preprocessed=True
):
    """
    :param img_size: The train_x pictures will be img_size x img_size pixels
    :param preprocessed: True if we should use preprocessed data (noise removed), False if raw data
    :return: train_x, train_y which are the input examples train_x and their labels train_y to use for training
    """
    global LABEL_TO_INTEGER, LABELS

    labels = np.genfromtxt('../data/train_labels.csv', names=True, delimiter=',',
                           dtype=[('Id', 'i8'), ('Category', 'S5')])
    num_data = labels.shape[0]

    if preprocessed:
        raw_data = np.array(pickle.load(open("../data/train_set.p", "rb")))
        images = np.array([
            # Adding zeros where there are missing pixels to fill each picture up to 100x100 pixels
            np.pad(image, ((0, img_size - image.shape[0]), (0, img_size - image.shape[1])), 'constant')
            for category in raw_data for image in category
        ])
        # Needed until the preprocessed data matches the labels count (we're missing one class)
        num_data = images.shape[0]
    else:
        images = np.load('../data/train_images.npy', encoding='latin1')

    # Set the global variables
    LABELS = list(set([labels[i][1] for i in range(0, num_data)]))
    # Needed until the preprocessed data matches the labels count (we're missing one class)
    if preprocessed:
        LABELS.remove(b'squig')
    for i in range(0, len(LABELS)):
        LABEL_TO_INTEGER[LABELS[i]] = i

    data = [
        ((images[i]).reshape(img_size, img_size, 1), label_to_onehot(labels[i][1])) for i in range(0, num_data)
    ]
    random.shuffle(data)

    inputs, outputs = zip(*data)
    return np.array(inputs), np.array(outputs)


def label_to_onehot(label_string):
    result = np.zeros(len(LABELS))
    result[LABEL_TO_INTEGER[label_string]] = 1
    return result


def int_to_label(label_integer):
    return LABELS[label_integer]


# def load_train_labels():
#     raw_data = np.genfromtxt('../data/train_labels.csv', names=True, delimiter=',',
#                              dtype=[('Id', 'i8'), ('Category', 'S5')])
#     return raw_data


def load_test_data():
    raw_data = np.load('../data/test_images.npy', encoding='latin1')
    #return [(x[1]).reshape(100, 100) for x in raw_data]
    random.shuffle(raw_data)
    return raw_data


def create_convnet(
    learning_rate,
    num_categories,
    img_size,
    cnn_engine
):
    convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

    if cnn_engine == 0:
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, num_categories, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy',
                             name='target')

    if cnn_engine == 1:
        convnet = conv_2d(convnet, 32, 3, activation='relu', regularizer="L2")
        convnet = max_pool_2d(convnet, 2)
        convnet = local_response_normalization(convnet)
        convnet = conv_2d(convnet, 64, 3, activation='relu', regularizer="L2")
        convnet = max_pool_2d(convnet, 2)
        convnet = local_response_normalization(convnet)
        convnet = fully_connected(convnet, 128, activation='tanh')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 256, activation='tanh')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, num_categories, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy', name='target')

    if cnn_engine == 2:
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, num_categories, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy',
                             name='target')

    if cnn_engine == 3:
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 128, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 256, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 512, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 1024, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, num_categories, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy',
                             name='target')

    if cnn_engine == 4:
        for i in range(0, 3):
            convnet = conv_2d(convnet, 32, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)
            convnet = conv_2d(convnet, 64, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, num_categories, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy',
                             name='target')

    return tflearn.DNN(convnet, tensorboard_dir='log')
