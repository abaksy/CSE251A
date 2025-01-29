from typing import List
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NearestNeighbours:
    """
    Implements 1-Nearest neighbours algorithm
    """

    def __init__(self):
        self.k = 1
        self.train_imgs = None
        self.train_labels = None

    def fit(self, train_imgs: List, train_labels: List):
        """
        "Train" the KNN classifier, i.e. save the training dataset into memory
        """
        self.train_imgs = train_imgs
        self.train_labels = train_labels

    def predict(self, test_imgs):
        """
        Predict the label of all the images in `test_imgs` using KNN classification

        Arguments:
        `test_imgs`: A `list` of `np.array`s of size (784,)
        """
        dists = euclidean_distances(test_imgs, self.train_imgs)
        min_idx = np.argmin(dists, axis=1)
        return self.train_labels[min_idx]

    def accuracy(self, test_labels, true_labels):
        """
        Return accuracy computed from the test labels and the true labels

        Arguments:
        `test_labels`: Labels returned by classifier
        `true_labels`: Ground truth labels used to compute accuracy
        """
        assert len(test_labels) == len(true_labels)
        N = len(test_labels)
        correct = 0

        for y, y_hat in zip(test_labels, true_labels):
            correct += y == y_hat

        return correct / N
