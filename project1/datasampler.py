from abc import abstractmethod, ABC
import random
from collections import Counter


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

    @abstractmethod
    def sample_data(self, M: int, x_train: list, y_train: list):
        pass


class RandomSampler(DataSampler):
    """
    Randomly sample examples from the dataset and return the samples
    """

    def __init__(self, M: int):
        super().__init__(M)

    def sample_data(self, x_train: list, y_train: list):
        """
        Randomly sample examples from the dataset and return the samples, using `random.sample`
        """
        assert len(x_train) == len(y_train)
        assert self.M <= len(x_train)
        train_samples = random.sample(list(zip(x_train, y_train)), self.M)
        x_train_samples = [x[0] for x in train_samples]
        y_train_samples = [x[1] for x in train_samples]
        return (x_train_samples, y_train_samples)


class RandomClassSampler(DataSampler):
    """
    Randomly sample M/N examples from each class of the data and return the samples,
    where N is the number of classes in the categorical data
    """

    def __init__(self, M: int):
        super().__init__(M)

    def sample_data(self, x_train: list, y_train: list):
        """
        Randomly sample M/N examples from each class of the data and return the samples,
        where N is the number of classes in the categorical data
        """
        assert len(x_train) == len(y_train)
        assert self.M <= len(x_train)

        classes = set(y_train)
        N = len(classes)
        train_samples = list()
        for cl in classes:
            filtered = [(x, y) for x, y in zip(x_train, y_train) if y == cl]
            train_samples.extend(random.sample(filtered, self.M // N))
        x_train_samples = [x[0] for x in train_samples]
        y_train_samples = [x[1] for x in train_samples]
        return (x_train_samples, y_train_samples)


class ProportionalRandomClassSampler(DataSampler):
    r"""
    Randomly sample k_i samples from class i, such that `\sum k_i = M` and
    the frequency of samples is proportional to their frequency in the original dataset
    """

    def __init__(self, M: int):
        super().__init__(M)

    def sample_data(self, x_train: list, y_train: list):
        """ """
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

        train_samples = list()
        for cl in classes:
            filtered = [(x, y) for x, y in zip(x_train, y_train) if y == cl]
            train_samples.extend(random.sample(filtered, sample_freq[cl]))
        x_train_samples = [x[0] for x in train_samples]
        y_train_samples = [x[1] for x in train_samples]
        return (x_train_samples, y_train_samples)
