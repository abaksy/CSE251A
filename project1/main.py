from dataloader import MNISTDataLoader
from knn import KNearestNeighbours
from datasampler import RandomSampler, RandomClassSampler, ProportionalRandomClassSampler
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
    print("Before sampling:")
    print("TRAIN:", len(x_train), len(y_train))

    print("TEST:", len(x_test), len(y_test))

    rs = ProportionalRandomClassSampler(1000)

    x_train_1, y_train_1 = rs.sample_data(x_train, y_train)
    
    print("After sampling:")
    print("TRAIN:", len(x_train_1), len(y_train_1))

    model = KNearestNeighbours(1)
    model.fit(x_train_1, y_train_1)

    start = time.perf_counter()

    y_hat = model.predict(x_test)

    end = time.perf_counter()

    acc = model.accuracy(y_hat, y_test)

    print("Accuracy of 1-NN Model:", acc, " time taken", end-start)

