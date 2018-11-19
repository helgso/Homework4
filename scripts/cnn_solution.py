import os

from scripts import cnn_utils


def main():
    learning_rate = 1e-3
    num_categories = 31
    epochs = 3
    img_size = 100
    cnn_engine = 4
    models_folder = 'cnn-saved-models'
    model_name = 'kaggle-{}-{}.model'.format(learning_rate, str(cnn_engine))

    model = cnn_utils.create_convnet(learning_rate, num_categories, img_size, cnn_engine)

    if os.path.exists('{}/{}.meta'.format(models_folder, model_name)):
        model.load(model_name)
        print('model loaded from a previous training. Can be used for predictions')

        # Todo: Write prediction code
    else:
        # Loading in the training dataset
        train_x, train_y = cnn_utils.load_training_dataset()
        # test_dataset = utils.load_test_data()

        # Splitting the training dataset into a training and a validation set
        valid_x, valid_y = train_x[-500:], train_y[-500:]
        train_x, train_y = train_x[:-500], train_y[:-500]

        # Traing the model
        model.fit(
            {'input': train_x}, {'target': train_y},
            n_epoch=epochs,
            validation_set=({'input': valid_x}, {'target': valid_y}),
            snapshot_step=500, show_metric=True, run_id=model_name
        )

        # Save for future predictions
        model.save('{}/{}'.format(models_folder, model_name))


if __name__ == '__main__':
    main()
