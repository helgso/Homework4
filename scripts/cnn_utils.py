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
    num_categories,
    noisy=False
):
    """
    Load the preprocessed training set (noise removed)

    :param img_size: The train_x pictures will be img_size * img_size pixels
    :param num_categories: How many categories we want to predict
    :param noisy: True if we should load the noisy training set, False if we want to load the denoised training set
    :return: train_x, train_y which are the input examples train_x and their labels train_y to use for training
    """
    raw_images = np.array(pickle.load(open("../data/train_set_{}.p".format(num_categories), "rb")))
    raw_labels = np.array(pickle.load(open("../data/cat_examples_full_names.p", "rb")))

    images = np.array([
        # Adding zeros where there are missing pixels to fill each picture up to 100x100 pixels
        pad_image_up_to(image, img_size)
        for category in raw_images for image in category
    ])
    labels = np.array([
        [raw_labels[i]] * len(raw_images[i]) for i in range(0, num_categories)
    ])
    labels = np.array([label for category in labels for label in category])

    num_data = images.shape[0]
    set_labels_global_variables(raw_labels, num_categories)

    data = [
        ((images[i]).reshape(img_size, img_size, 1), label_to_onehot(labels[i])) for i in range(0, num_data)
    ]
    random.shuffle(data)

    inputs, outputs = zip(*data)
    return np.array(inputs), np.array(outputs)


def set_labels_global_variables(
    raw_labels,
    num_categories
):
    global LABELS, LABEL_TO_INTEGER

    # Set the global variables
    LABELS = raw_labels[:num_categories]
    for i in range(0, len(LABELS)):
        LABEL_TO_INTEGER[LABELS[i]] = i


def label_to_onehot(
    label_string
):
    result = np.zeros(len(LABELS))
    result[LABEL_TO_INTEGER[label_string]] = 1
    return result


def int_to_label(
    label_integer
):
    return LABELS[label_integer]


def load_testing_dataset(
    img_size,
    num_categories,
    noisy=False
):
    """
    Load the preprocessed test set (noise removed)

    :param img_size: The test_x pictures will be img_size * img_size pixels
    :param num_categories: How many categories we want to predict
    :param noisy: True if we should load the noisy testing set, False if we want to load the denoised testing set
    :return: test_x which are the input examples whose classification we use to submit to Kaggle
    """
    raw_images = np.array(pickle.load(open("../data/test_set_v1.p", "rb")))

    images = np.array([
        # Adding zeros where there are missing pixels to fill each picture up to 100x100 pixels
        # and reshaping for tflearn
        (pad_image_up_to(image, img_size)).reshape(img_size, img_size, 1)
        for image in raw_images
    ])

    raw_labels = np.array(pickle.load(open("../data/cat_examples_full_names.p", "rb")))
    set_labels_global_variables(raw_labels, num_categories)

    return images


def pad_image_up_to(
    image,
    img_size
):
    """
    Preserve the scale of the image image but pad it with zeros up to the dimensions img_size * img_size

    :param image: The image to pad
    :param img_size: The size of the image to pad up to
    :return: The padded image
    """
    return np.pad(image, ((0, img_size - image.shape[0]), (0, img_size - image.shape[1])), 'constant')


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

    if cnn_engine == 5:
        convnet = conv_2d(convnet, 32, 3, activation='relu', regularizer="L2")
        convnet = max_pool_2d(convnet, 2)
        convnet = local_response_normalization(convnet)
        convnet = conv_2d(convnet, 64, 3, activation='relu', regularizer="L2")
        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = local_response_normalization(convnet)
        convnet = fully_connected(convnet, 128, activation='tanh')
        convnet = dropout(convnet, 0.5)
        convnet = fully_connected(convnet, 512, activation='tanh')
        convnet = dropout(convnet, 0.5)
        convnet = fully_connected(convnet, 31, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy', name='target')



    return tflearn.DNN(convnet, tensorboard_dir='log')
