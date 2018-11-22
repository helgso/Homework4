import os

from scripts import cnn_utils
import numpy as np


def main():
    learning_rate = 3e-5
    num_categories = 30
    epochs = 50
    img_size = 100
    cnn_engine = 4
    noisy = False
    validation_set_size = 0.05
    models_folder = '../results/cnn/saved-models'
    model_name = 'cnn-{}-{}-{}-{}-{}'.format(num_categories, learning_rate, validation_set_size, noisy, cnn_engine)
    model_file_path = '{}/{}'.format(models_folder, model_name)
    predictions_folder = '../results/cnn'
    predictions_file_path = '{}/{}.csv'.format(predictions_folder, model_name)

    model = cnn_utils.create_convnet(learning_rate, num_categories, img_size, cnn_engine)

    if os.path.exists('{}.meta'.format(model_file_path)):
        model.load(model_file_path)
        print('Loaded {}.meta from previous training. Do you want to continue training or predict? (train/predict)'.format(model_file_path))
        value = input()

        if value.lower() == 'predict' or value.lower() == 'p':
            predict(model, predictions_file_path, img_size, num_categories)
            return

    train(model, model_file_path, epochs, img_size, num_categories, validation_set_size, noisy)


def train(
    model,
    model_file_path,
    epochs,
    img_size,
    num_categories,
    validation_set_size,
    noisy
):
    """
    Given a model, train it and save it as the file model_file_path

    :param model: A tflearn model
    :param model_file_path: The file path where we will save our model after training
    :param img_size: The size we want of the images of our training set
    :param num_categories: How many categories we want to predict
    :param validation_set_size: Percentage of training set we will use for validation during training
    :param noisy: Whether we use the noisy training set for training or not
    """
    train_x, train_y = cnn_utils.load_training_dataset(img_size, num_categories, noisy)

    valid_length = int(len(train_x)*validation_set_size)

    # Splitting the training dataset into a training and a validation set
    valid_x, valid_y = train_x[-valid_length:], train_y[-valid_length:]
    train_x, train_y = train_x[:-valid_length], train_y[:-valid_length]

    # Training the model
    model.fit(
        {'input': train_x}, {'target': train_y},
        n_epoch=epochs,
        validation_set=({'input': valid_x}, {'target': valid_y}), batch_size=32,
        snapshot_step=500, show_metric=True, run_id=os.path.basename(model_file_path)
    )

    print('Training complete.')
    print('Saving the trained model at {}.meta for future use'.format(model_file_path))
    model.save(model_file_path)


def predict(
    model,
    predictions_file_path,
    img_size,
    num_categories
):
    """
    Given a model, create a predictions file at predictions_file_path for the preprocessed test set

    :param model: A tflearn model
    :param predictions_file_path: The file path of our predictions for the model on the test data
    :param img_size: The size we want of the images of our test set
    :param num_categories: How many categories we want to predict
    """
    test_x = cnn_utils.load_testing_dataset(img_size, num_categories)

    print("Predicting classes ...")

    # We do predictions in batches to lower the computational complexity (I got an out of memory exception when
    # trying to predict all 10000 testing examples at once (Helgi)
    batch_size = 500
    results = []
    for i in range(0, len(test_x), batch_size):
        raw_predictions = model.predict(test_x[i:i+batch_size])
        predictions = [[j, cnn_utils.int_to_label(np.argmax(raw_predictions[j%batch_size]))] for j in range(len(results), len(results)+batch_size)]
        results += predictions
        print(i)

    print("Done. Saving predictions at {}".format(predictions_file_path))
    np.savetxt(predictions_file_path, [['Id', 'Category']] + results, delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
