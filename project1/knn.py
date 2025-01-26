from typing import List
import numpy as np

class KNearestNeighbours:
    '''
    Implements K-Nearest neighbours algorithm

    Arguments:

    `k`: Number of nearest neighbours to consider when classifying a new instance
    '''
    def __init__(self, k: int = 1):
        self.k = k
        self.train_imgs = None
        self.train_labels = None
       
    def distance(self, x1, x2):
        '''
        Return Euclidean disance between x1 and x2

        Arguments:
        `x1`, `x2`: Arrays of type `np.array`
        '''
        return np.linalg.norm(x1 - x2)

    def fit(self, train_imgs: List, train_labels: List):
        '''
        "Train" the KNN classifier, i.e. save the training dataset into memory
        '''
        self.train_imgs = train_imgs
        self.train_labels = train_labels

    def predict_instance(self, test_instance):
        '''
        For a single test instance, return its label by getting the nearest neighbours and computing the label

        Arguments:
        `test_instance`: A single instance, represented by an `np.array` of size 28x28
        '''
        min_dist = 100000000000
        test_label = -1
        for img, label in zip(self.train_imgs, self.train_labels):
            # if not np.array_equal(test_instance, img):
            dist = self.distance(test_instance, img)
            if dist < min_dist:
                min_dist = dist
                test_label = label
        
        return (test_label, min_dist)

    def predict(self, test_imgs):
        '''
        Predict the label of all the images in `test_imgs` using KNN classification

        Arguments:
        `test_imgs`: A `list` of `np.array`s of size 28 x 28
        '''
        labels = list()
        for i, sample in enumerate(test_imgs):
            label, _ = self.predict_instance(sample)
            labels.append(label)
        return labels
    
    def accuracy(self, test_labels, true_labels):
        '''
        Return accuracy computed from the test labels and the true labels

        Arguments:
        `test_labels`: Labels returned by classifier
        `true_labels`: Ground truth labels used to compute accuracy
        '''
        assert len(test_labels) == len(true_labels)
        N = len(test_labels)
        correct = 0

        for y, y_hat in zip(test_labels, true_labels):
            correct += (y == y_hat)
        
        return correct/N
