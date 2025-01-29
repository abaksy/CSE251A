from abc import abstractmethod, ABC
import random
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class DataSampler(ABC):
    """
    Abstract base class for all data sampling methods

    All data samplers must implement the `sample_data` method
    that takes in the desired number of samples `M`
    as well as a train and test set.
    """

    def __init__(self, M: int):
        super().__init__()
        self.M = M
        self.name = ""

    @abstractmethod
    def sample_data(self, x_train: list, y_train: list):
        pass


class RandomSampler(DataSampler):
    """
    Randomly sample examples from the dataset and return the samples
    """

    def __init__(self, M: int):
        super().__init__(M)
        self.name = "RandomSampler"

    def sample_data(self, x_train: list, y_train: list):
        """
        Randomly sample examples from the dataset and return the samples, using `random.sample`
        """
        assert len(x_train) == len(y_train)
        assert self.M <= len(x_train)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        indices = np.random.choice(x_train.shape[0], self.M, replace=False)
        x_train_samples = x_train[indices]
        y_train_samples = y_train[indices]
        return (x_train_samples, y_train_samples)


class RandomClassSampler(DataSampler):
    """
    Randomly sample M/N examples from each class of the data and return the samples,
    where N is the number of classes in the categorical data
    """

    def __init__(self, M: int):
        super().__init__(M)
        self.name = "RandomClassSampler"

    def sample_data(self, x_train: list, y_train: list):
        """
        Randomly sample M/N examples from each class of the data and return the samples,
        where N is the number of classes in the categorical data
        """
        assert len(x_train) == len(y_train)
        assert self.M <= len(x_train)

        classes = set(y_train)
        N = len(classes)
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_train_samples = list()
        y_train_samples = list()

        for cl in classes:
            class_idx = np.where(y_train == cl)
            x = x_train[class_idx]
            y = y_train[class_idx]

            sample_idx = np.random.choice(x.shape[0], self.M // N, replace=False)
            x_train_samples.append(x[sample_idx])
            y_train_samples.append(y[sample_idx])

        return (np.concatenate(x_train_samples), np.concatenate(y_train_samples))


class ProportionalRandomClassSampler(DataSampler):
    r"""
    Randomly sample k_i samples from class i, such that `\sum k_i = M` and
    the frequency of samples is proportional to their frequency in the original dataset
    """

    def __init__(self, M: int):
        super().__init__(M)
        self.name = "PropRandomSampler"

    def sample_data(self, x_train: list, y_train: list):
        """ """
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        assert len(x_train) == len(y_train)
        assert self.M <= len(x_train)

        classes = set(y_train)
        dataset_freq = Counter(y_train)
        sample_freq = {
            c: round((dataset_freq[c] / len(y_train)) * self.M) for c in dataset_freq
        }
        num_samples = sum(sample_freq.values())

        if num_samples < self.M:
            difference = self.M - num_samples
            most_frequent_class = max(dataset_freq.items(), key=lambda x: x[1])[0]
            sample_freq[most_frequent_class] += difference

        x_train_samples = list()
        y_train_samples = list()

        for cl in classes:
            class_idx = np.where(y_train == cl)
            x = x_train[class_idx]
            y = y_train[class_idx]

            sample_idx = np.random.choice(x.shape[0], sample_freq[cl], replace=False)
            x_train_samples.append(x[sample_idx])
            y_train_samples.append(y[sample_idx])

        return (np.concatenate(x_train_samples), np.concatenate(y_train_samples))


class KMeansSampler(DataSampler):
    def __init__(self, M: int):
        super().__init__(M)
        self.name = "KMeansSampler"

    def sample_data(self, x_train: list, y_train: list):
        assert len(x_train) == len(y_train)
        assert self.M <= len(x_train)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        cluster_model = KMeans(n_clusters=self.M)

        cluster_model.fit(x_train, y_train)
        # Get distances from each point to each centroid
        distances = euclidean_distances(x_train, cluster_model.cluster_centers_)

        # For each cluster, find the point closest to its centroid
        sampled_indices = []
        for cluster_idx in range(self.M):
            # Get points assigned to this cluster
            cluster_points = np.where(cluster_model.labels_ == cluster_idx)[0]
            if cluster_points.shape[0] == 0:
                continue
            # Find the point closest to the centroid
            closest_point_idx = cluster_points[
                np.argmin(distances[cluster_points, cluster_idx])
            ]
            sampled_indices.append(closest_point_idx)

        # Convert to numpy array for easier indexing
        sampled_indices = np.array(sampled_indices)

        return x_train[sampled_indices], y_train[sampled_indices]


class HierarchicalKMeansSampler(DataSampler):
    def __init__(self, M: int):
        super().__init__(M)
        self.name = "HierarchicalKMeansSampler"

    def sample_data(self, x_train, y_train):
        assert len(x_train) == len(y_train)
        assert self.M <= len(x_train)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        classes = set(y_train)
        n_classes = len(classes)

        x_samples = list()
        y_samples = list()

        for c in classes:
            indices = np.where(y_train == c)
            x = x_train[indices]
            y = y_train[indices]

            K = self.M // n_classes
            cluster_model = KMeans(K)
            cluster_model.fit(x, y)

            sampled_indices = []
            distances = euclidean_distances(x, cluster_model.cluster_centers_)
            for cluster_idx in range(K):
                # Get points assigned to this cluster
                cluster_points = np.where(cluster_model.labels_ == cluster_idx)[0]
                if cluster_points.shape[0] == 0:
                    continue
                # Find the point closest to the centroid
                closest_point_idx = cluster_points[
                    np.argmin(distances[cluster_points, cluster_idx])
                ]
                sampled_indices.append(closest_point_idx)

            # Convert to numpy array for easier indexing
            sampled_indices = np.array(sampled_indices)
            x_samples.append(x[sampled_indices])
            y_samples.append(y[sampled_indices])

        return np.concatenate(x_samples), np.concatenate(y_samples)


class StratifiedKMeansSampler(DataSampler):
    """
    Split the dataset into 5 parts, each part has prop number of samples of all classes

    From each part, sample M/5 samples using k means
    """

    def __init__(self, M):
        super().__init__(M)
        self.name = "StratifiedKMeansSampler"

    def sample_data(self, x_train, y_train):
        assert len(x_train) == len(y_train)
        assert self.M <= len(x_train)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        classes = set(y_train)

        x_samples = list()
        y_samples = list()

        dataset_freq = Counter(y_train)
        sample_freq = {
            c: round((dataset_freq[c] / len(y_train)) * self.M) for c in dataset_freq
        }
        num_samples = sum(sample_freq.values())

        if num_samples < self.M:
            difference = self.M - num_samples
            most_frequent_class = max(dataset_freq.items(), key=lambda x: x[1])[0]
            sample_freq[most_frequent_class] += difference

        for c in classes:
            indices = np.where(y_train == c)
            x = x_train[indices]
            y = y_train[indices]

            K = sample_freq[c]
            cluster_model = KMeans(K)
            cluster_model.fit(x, y)

            sampled_indices = []
            distances = euclidean_distances(x, cluster_model.cluster_centers_)
            for cluster_idx in range(K):
                # Get points assigned to this cluster
                cluster_points = np.where(cluster_model.labels_ == cluster_idx)[0]
                if cluster_points.shape[0] == 0:
                    continue
                # Find the point closest to the centroid
                closest_point_idx = cluster_points[
                    np.argmin(distances[cluster_points, cluster_idx])
                ]
                sampled_indices.append(closest_point_idx)

            # Convert to numpy array for easier indexing
            sampled_indices = np.array(sampled_indices)
            x_samples.append(x[sampled_indices])
            y_samples.append(y[sampled_indices])

        return np.concatenate(x_samples), np.concatenate(y_samples)
