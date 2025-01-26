from dataloader import MNISTDataLoader
from knn import KNearestNeighbours
from datasampler import *
import os
from testbench import TestBench
import pprint

if __name__ == '__main__':
    base_path = 'dataset'
    training_images_filepath = os.path.join(base_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(base_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(base_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(base_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


    data_loader = MNISTDataLoader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (x_train, y_train), (x_test, y_test) = data_loader.read_data()

    model = KNearestNeighbours(1)

    Ms = [1000, 5000, 10_000, 20_000, 30_000]
    samplers = [RandomSampler, RandomClassSampler, ProportionalRandomClassSampler, KMeansSampler, HierarchicalKMeansSampler]

    for sampler in samplers:
        for M in Ms:
            rs = sampler(M)
            print(f"Testing {rs.name} sampler with {M} samples")
            test_bench = TestBench(model, rs)
            print("Running inference on test set!")
            results = test_bench.run_pipeline(30, x_train, y_train, x_test, y_test)
            pprint.pprint(results[1])
