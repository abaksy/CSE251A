from datasampler import DataSampler
from knn import KNearestNeighbours
import time


class TestBench:
    def __init__(self, clf: KNearestNeighbours = None, sampler: DataSampler = None):
        self.clf = clf
        self.sampler = sampler

    def sample_train_test_pipeline(self, x_train, y_train, x_test, y_test):
        """
        Executes one iteration of the sample->train->test pipeline
        
        Warning: this method should be accessed from the `run_pipeline` method only, 
        DO NOT call this directly
        """
        if self.sampler is not None:
            x_train_sample, y_train_sample = self.sampler.sample_data(x_train, y_train)
        else:
            x_train_sample = x_train
            y_train_sample = y_train
            print(
                "Warning: sampler is None, using entire train set for classification!"
            )

        if self.clf is not None:
            self.clf.fit(x_train_sample, y_train_sample)
            start = time.perf_counter()
            y_hat = self.clf.predict(x_test)
            end = time.perf_counter()

            acc = self.clf.accuracy(y_hat, y_test)
            time_taken = end - start

        else:
            acc = 0.0
            time_taken = 0.0
            print("Warning: model is None, not running classifier!")
        return acc, time_taken

    def run_pipeline(self, N: int, x_train, y_train, x_test, y_test):
        """
        Run the entire sample->train->test pipeline `N` times,
        returning the accuracy and time taken for inference on the test set
        """
        results = list()
        for i in range(N):
            print(f"running iteration {i+1} of model....")
            acc, time_taken = self.sample_train_test_pipeline(
                x_train, y_train, x_test, y_test
            )
            results.append((acc, time_taken))

        return results
