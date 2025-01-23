import struct
from array import array
import numpy as np


class MNISTDataLoader:
    """
    Class that loads MNIST data and returns the data split into training and test sets

    Arguments :

    `train_imgs_path`: Path to train images folder
    `train_labels_path`: Path to train labels file
    `test_imgs_path`: Path to test images folder
    `test_labels_path`: Path to test labels file
    """

    def __init__(
        self,
        train_imgs_path: str,
        train_labels_path: str,
        test_imgs_path: str,
        test_labels_path: str,
    ):
        self.train_imgs_path = train_imgs_path
        self.train_labels_path = train_labels_path
        self.test_imgs_path = test_imgs_path
        self.test_labels_path = test_labels_path

    def read_images_labels(self, images_filepath, labels_filepath):
        '''
        Read images and labels and return as arrays
        '''
        labels = list()
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append(np.zeros(rows*cols))
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            # img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def read_data(self):
        '''
        Public method to read MNIST data and return both train and test sets as arrays
        '''
        (train_imgs, train_labels) = self.read_images_labels(
            self.train_imgs_path, self.train_labels_path
        )

        (test_imgs, test_labels) = self.read_images_labels(
            self.test_imgs_path, self.test_labels_path
        )

        return (train_imgs, train_labels), (test_imgs, test_labels)
