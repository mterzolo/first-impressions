import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """
    Generates batches of image data for keras
    """
    def __init__(self, partition, list_IDs, labels, batch_size=32, dim=(20, 224, 224), n_channels=3,
                 shuffle=True):
        """
        Initialization
        :param partition: which data set
        :param list_IDs: list of ID numbers that correspond to each video
        :param labels: scores for the corresponding videos
        :param batch_size: how many observations to be fed through the network at a time
        :param dim: dimensions for the observations
        :param n_channels: number of RBG channels for the images
        :param shuffle: randomly shuffle the data set in between batches
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.partition = partition
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: integer
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: position of the video in the array
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return: new index
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples
        :param indexes: positions of the samples to take for the batch
        :return: batch for both target and features
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size)

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            X[i, ] = np.load('../data/image_data/npy_files/{}_data/{}.npy'.format(self.partition, ID))

            # Store class
            y[i] = self.labels[ID]

        return X, y