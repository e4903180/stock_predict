import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

class LoadData:
    def __init__(self, total_windows, window_length):
        """Initializes the LoadData class.

        Args:
            total_windows (int): The total number of windows.
            window_length (int): The length of each window.
        """

        self.train_data = np.ndarray(shape=(total_windows, window_length), dtype=float)
        self.test_data = np.ndarray(shape=(total_windows, window_length), dtype=float)
        self.date_data_start_list = []
        self.date_predict_start_list = []
        self.date_predict_end_list = []
        self.test_data_start_list = []

    def load_and_split_data(self, stock_name, date_predict_start, window_length, slide_range, total_windows):
        """Loads and splits the data.

        Args:
            stock_name (str): The stock name.
            date_predict_start (str): The start date for prediction.
            window_length (int): The length of each window.
            slide_range (int): The slide range.
            total_windows (int): The total number of windows.

        Returns:
            train_data: numpy.ndarray
                The training data arrays.
                shape: (number of windows, window_length)
            test_data: numpy.ndarray
                The testing data arrays.
                shape: (number of windows, window_length)
        """

        all_data = yf.Ticker(stock_name).history(period='max')
        all_data.index = all_data.index.date
        # all_data['Close'].iloc[0:len(all_data['Close'])] = list(range(0, len(all_data['Close'])))
        date_predict_start = self._check_start(date_predict_start, all_data)
        predict_start = all_data.index.get_loc(date_predict_start)
        for i in range(total_windows):
            predict_end = predict_start + window_length
            data_start = predict_start - window_length
            self.train_data[i, :] = all_data['Close'].iloc[data_start:predict_start].values
            self.test_data[i, :] = all_data['Close'].iloc[predict_start:predict_end].values
            data_start = data_start + slide_range
            predict_start = predict_start + slide_range
        return self.train_data, self.test_data

    def _check_start(self, date_predict_start, all_data):
        """Checks the start date for prediction.

        Args:
            date_predict_start (str): The start date for prediction.
            all_data (pandas.DataFrame): The complete data.

        Returns:
            datetime.date: The validated start date for prediction.
        """

        date_predict_start = datetime.datetime.strptime(
            date_predict_start, '%Y-%m-%d').date()
        check = date_predict_start in all_data.index
        while (check == False):
            date_predict_start = date_predict_start + \
                relativedelta(days=+1)
            check = date_predict_start in list(all_data.index)
        return date_predict_start


if __name__ == '__main__':
    stock_name = "^GSPC"
    date_predict_start = '2020-01-01'
    window_length = 10
    slide_range = 10
    total_windows = 5
    dataloader = LoadData(total_windows, window_length)
    train_data, test_data =\
        dataloader.load_and_split_data(stock_name, date_predict_start, window_length, slide_range, total_windows)
    print(train_data)
    print(train_data.shape)