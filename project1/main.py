from dataloader import MNISTDataLoader
from knn import KNearestNeighbours
import os
import time

if __name__ == '__main__':
    base_path = 'dataset'
    training_images_filepath = os.path.join(base_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(base_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(base_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(base_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


    data_loader = MNISTDataLoader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (x_train, y_train), (x_test, y_test) = data_loader.read_data()

    print("TRAIN:", len(x_train), len(y_train))

    print("TEST:", len(x_test), len(y_test))

    model = KNearestNeighbours(1)
    model.fit(x_train, y_train)

    start = time.perf_counter()

    y_hat = model.predict(x_test)

    end = time.perf_counter()

    acc = model.accuracy(y_hat, y_test)

    print("Accuracy with basic 1-NN:", acc, " time taken", end-start)

