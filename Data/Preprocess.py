import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Preprocess:
    """
    Class to preprocess training and testing data.

    Attributes:
        x_length (int): Length of the input sequence.
        y_length (int): Length of the target sequence.
    """

    def __init__(self, x_length, y_length):
        self.x_length = x_length
        self.y_length = y_length

    def preprocess_train_data(self, train_data):
        """
        Preprocess training data.

        Args:
            train_data (numpy.ndarray): Training data. Shape: (number of windows, window length, number of features)
            self.x_length (int): Length of the input sequence.
            self.y_length (int): Length of the target sequence.

        Returns:
            Tuple containing preprocessed input data, target data, and scaler object.
        """
        split_num = len(range(self.x_length, train_data.shape[1]-self.y_length+1))
        x_train = np.zeros((train_data.shape[0], split_num, self.x_length, train_data.shape[-1]))
        y_train = np.zeros((train_data.shape[0], split_num, self.y_length))
        scaler = MinMaxScaler(feature_range=(0, 1))
        for window_idx in range(train_data.shape[0]):
            for feature_idx in range(train_data.shape[-1]):
                scaled_data_train = scaler.fit_transform(train_data[window_idx, :, feature_idx].reshape(-1, 1)).flatten()
                split_count = 0
                for idx in range(self.x_length, len(scaled_data_train)-self.y_length+1):
                    x_train[window_idx, split_count, :, feature_idx] = scaled_data_train[idx-self.x_length:idx]
                    if feature_idx == 3:
                        y_train[window_idx, split_count] = scaled_data_train[idx:idx+self.y_length]
                    split_count += 1
        return x_train, y_train, scaler

    def preprocess_test_data(self, train_data, test_data):
        """
        Preprocess testing data.

        Args:
            train_data (numpy.ndarray): Data used for scaling. Shape: (number of windows, window length, number of features)
            test_data (numpy.ndarray): Testing data. Shape: (number of windows, window length, number of features)
            self.x_length (int): Length of the input sequence.
            self.y_length (int): Length of the target sequence.
            slide (int): Sliding window step size.

        Returns:
            Tuple containing preprocessed input data, target data, and scaler object.
        """
        slide = self.y_length
        split_num = len(range(train_data.shape[1], train_data.shape[1]+test_data.shape[1]-self.y_length+1, slide))
        x_test = np.zeros((test_data.shape[0], split_num, self.x_length, test_data.shape[-1]))
        y_test = np.zeros((test_data.shape[0], split_num, self.y_length))
        scaler = MinMaxScaler(feature_range=(0, 1))
        concat_data = np.concatenate([train_data, test_data], axis=1)
        for window_idx in range(test_data.shape[0]):
            for feature_idx in range(test_data.shape[-1]):
                scaled_data = scaler.fit_transform(concat_data[window_idx, :, feature_idx].reshape(-1, 1)).flatten()
                split_count = 0
                for idx in range(train_data.shape[1], len(scaled_data)-self.y_length+1, slide):
                    x_test[window_idx, split_count, :, feature_idx] = scaled_data[idx-self.x_length:idx]
                    if feature_idx == 3:
                        y_test[window_idx, split_count] = scaled_data[idx:idx+self.y_length]
                    split_count += 1
        return x_test, y_test
    
    def shuffle(self, x_train, y_train):
        shuffle_index = np.random.permutation(x_train.shape[1])
        x_train_shuffled = x_train[:, shuffle_index]
        y_train_shuffled = y_train[:, shuffle_index]
        return x_train_shuffled, y_train_shuffled
