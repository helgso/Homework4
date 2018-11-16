import numpy as np


def load_train_data():
    raw_data = np.load('../data/train_images.npy', encoding='latin1')
    return [(x[1]).reshape(100, 100) for x in raw_data]


def load_train_labels():
    raw_data = np.genfromtxt('../data/train_labels.csv', names=True, delimiter=',',
                             dtype=[('Id', 'i8'), ('Category', 'S5')])
    return [x[1] for x in raw_data]


def load_test_data():
    raw_data = np.load('../data/test_images.npy', encoding='latin1')
    return [(x[1]).reshape(100, 100) for x in raw_data]
