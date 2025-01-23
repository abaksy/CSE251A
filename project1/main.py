from dataloader import MNISTDataLoader
from knn import KNearestNeighbours
from datasampler import KMeansSampler
import os
from testbench import TestBench

if __name__ == '__main__':
    base_path = 'dataset'
    training_images_filepath = os.path.join(base_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(base_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(base_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(base_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


    data_loader = MNISTDataLoader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (x_train, y_train), (x_test, y_test) = data_loader.read_data()
    rs = KMeansSampler(1000)

    model = KNearestNeighbours(1)
    
    test_bench = TestBench(model, rs)

    results = test_bench.run_pipeline(1, x_train, y_train, x_test, y_test)

    for a, tt in results:
        print(f"Accuracy: {a}, Time Taken: {tt} s")

