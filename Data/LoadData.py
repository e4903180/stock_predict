import numpy as np
import pandas as pd
import yfinance as yf


class LoadData:
    """
    Class to load and split stock data.

    Attributes:
        stock_name (str): Name of the stock.
        window_length (int): Length of the window for splitting data.
        slide_range (int): Range of sliding window for splitting data.
        num_windows (int): Total number of windows for splitting data.
        features_num (int): Number of features to consider from the stock data.
        features (list): List of column names to consider from the stock data.
        train_data (numpy.ndarray): Array to store training data. Shape: (num_windows, window_length, features_num)
        train_data_index (numpy.ndarray): Array to store indices of training data. Shape: (num_windows, window_length)
        test_data (numpy.ndarray): Array to store testing data. Shape: (num_windows, window_length, features_num)
        test_data_index (numpy.ndarray): Array to store indices of testing data. Shape: (num_windows, window_length)
    """

    def __init__(self, stock_name, window_length, slide_range, num_windows, feature_num=1):
        self.stock_name = stock_name
        self.window_length = int(window_length)
        self.slide_range = int(slide_range)
        self.num_windows = int(num_windows)
        self.features_num = feature_num - 1 if feature_num != 1 else 1
        self.features = ['Close', 'Open', 'High', 'Low', 'Volume']
        self.train_data = np.zeros(shape=(num_windows, window_length, self.features_num), dtype=float)
        self.train_data_index = np.zeros(shape=(num_windows, window_length), dtype=object)
        self.test_data = np.zeros(shape=(num_windows, window_length, self.features_num), dtype=float)
        self.test_data_index = np.zeros(shape=(num_windows, window_length), dtype=object)

    def load_and_split_data(self, date_predict_start):
        """
        Load and split data into training and testing sets.

        Args:
            date_predict_start (str): Start date for prediction.

        Returns:
            Tuple containing training data, testing data, training data indices, testing data indices, and all data.
        """
        all_data = yf.Ticker(self.stock_name).history(period='max').drop(columns=['Dividends', 'Stock Splits'])
        all_data.index = all_data.index.date
        date_predict_start = self._check_start(date_predict_start, all_data)
        predict_start = all_data.index.get_loc(date_predict_start)
        for idx in range(self.num_windows):
            predict_end = predict_start + self.window_length
            data_start = predict_start - self.window_length
            self.train_data[idx, :] = all_data[self.features[:self.features_num]].iloc[data_start:predict_start].values
            self.train_data_index[idx, :] = all_data.iloc[data_start:predict_start].index
            self.test_data[idx, :] = all_data[self.features[:self.features_num]].iloc[predict_start:predict_end].values
            self.test_data_index[idx, :] = all_data.iloc[predict_start:predict_end].index
            predict_start += self.slide_range
        return self.train_data, self.test_data, self.train_data_index, self.test_data_index, all_data

    def _check_start(self, date_predict_start, all_data):
        """
        Check if the prediction start date is valid.

        Args:
            date_predict_start (str): Start date for prediction.
            all_data (pd.DataFrame): All stock data.

        Returns:
            Valid start date for prediction.
        """
        date_predict_start = pd.to_datetime(date_predict_start)
        valid_date = all_data.index.asof(date_predict_start)
        return valid_date
