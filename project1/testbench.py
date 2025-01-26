from datasampler import DataSampler
from knn import KNearestNeighbours
import time
import numpy as np
from scipy import stats
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


class TestBench:
    def __init__(self, clf: KNearestNeighbours = None, sampler: DataSampler = None):
        self.clf = clf
        self.sampler = sampler
        self.result_dir = "results"
        self.plots_dir = "plots"
        self.file_name = f"{self.sampler.name}_{self.sampler.M}"

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

    def calculate_summary_stats(self, results: list, ci=0.95):
        """
        Return summary statistics from the accuracy metrics
        """
        results = np.array(results)[:, 0]
        mean = np.mean(results)
        stdev = np.std(results)

        n = len(results)
        t_value = stats.t.ppf((1 + ci) / 2, df=n - 1)
        margin_of_error = t_value * (stdev / np.sqrt(n))
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error

        stats_dict = {
            "data": list(results),
            "mean": mean,
            "std": stdev,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": ci,
        }

        return stats_dict

    def plot_data(self, statistics: dict):
        """
        Plot boxplot of accuracy data with iterations
        """
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        results = statistics['data']
        ci_lower = statistics['ci_lower']
        ci_upper = statistics['ci_upper']
        c_level = statistics['confidence_level']
        N = statistics['iters']
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        sns.boxplot(data=results, color='lightblue')
        
        # Add individual points
        sns.swarmplot(data=results, color='navy', alpha=0.5)
        
        # Add confidence interval
        plt.axhline(y=ci_lower, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=ci_upper, color='r', linestyle='--', alpha=0.5)
        
        # Customize plot
        # plt.title(f'Accuracy Distribution over {N} trials - \n{c_level*100}% Confidence Interval')
        plt.ylabel('Accuracy')
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()

        plt.savefig(os.path.join(self.plots_dir, self.file_name))

    def dump_file(self, stats: dict):
        """
        Write summary stats to file in JSON format
        """
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        string_data = json.dumps(stats)
        file_path = os.path.join(self.result_dir, self.file_name)

        with open(file_path, "w") as f:
            f.write(string_data)

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

        statistics = self.calculate_summary_stats(results)
        statistics["iters"] = N

        self.dump_file(statistics)
        self.plot_data(statistics)

        return results, statistics
