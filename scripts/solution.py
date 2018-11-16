import numpy as np
import matplotlib.pyplot as plt
import os

from scripts import utils


def main():
    # All three are of shape (10000, 1), each index corresponding to the 100x100 pixel image or label for that image
    images_train = utils.load_train_data()
    train_labels = utils.load_train_labels()
    test_data = utils.load_test_data()


if __name__ == '__main__':
    main()
