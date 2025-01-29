from dataloader import MNISTDataLoader
from knn import NearestNeighbours
from datasampler import *
import os
from testbench import TestBench

if __name__ == "__main__":
    base_path = "dataset"
    training_images_filepath = os.path.join(
        base_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
    )
    training_labels_filepath = os.path.join(
        base_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
    )
    test_images_filepath = os.path.join(
        base_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    )
    test_labels_filepath = os.path.join(
        base_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
    )

    data_loader = MNISTDataLoader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )

    (x_train, y_train), (x_test, y_test) = data_loader.read_data()
    print(x_train[0].shape, y_train[0])

    model = NearestNeighbours()

    Ms = [1000, 5000, 10000, 20000, 30000]
    samplers = [
        RandomSampler,]
    #     RandomClassSampler,
    #     ProportionalRandomClassSampler,
    #     KMeansSampler,
    #     HierarchicalKMeansSampler,
    #     StratifiedKMeansSampler,
    # ]

    for sampler in samplers:
        for M in Ms:
            rs = sampler(M)
            print(f"Testing {rs.name} sampler with {M} samples")
            test_bench = TestBench(model, rs)
            print("Running inference on test set!")
            results = test_bench.run_pipeline(30, x_train, y_train, x_test, y_test)
            print(f"Accuracy: {results[1]["mean"]}")
