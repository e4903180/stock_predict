# import yfinance as yf
# import datetime
# from dateutil.relativedelta import relativedelta
# from tqdm import tqdm
# import numpy as np

# class LoadData:

#     def __init__(self, n_slide, data_range) -> None:
#         self.train_data = np.ndarray(shape=(n_slide,data_range), dtype=float)
#         self.test_data = np.ndarray(shape=(n_slide,data_range), dtype=float)
#         self.date_data_start_list = []
#         self.date_predict_start_list = []
#         self.date_predict_end_list = []
#         self.test_data_start_list = []

#     def load_and_split_data(self, stock_name, date_predict_start, data_range, slide_range, n_slide):
#         all_data = yf.Ticker(stock_name).history(period='max')
#         all_data.index = all_data.index.date
#         # all_data['Close'].iloc[0:len(all_data['Close'])] = list(range(0, len(all_data['Close'])))
#         date_predict_start = self._check_start(date_predict_start, all_data)
#         predict_start = all_data.index.get_loc(date_predict_start)
#         for i in tqdm(range(n_slide)):
#             predict_end = predict_start + data_range
#             data_start = predict_start - data_range
#             self.train_data[i, :] = all_data['Close'].iloc[data_start:predict_start].values
#             self.test_data[i, :] = all_data['Close'].iloc[predict_start:predict_end].values
#             data_start = data_start + slide_range
#             predict_start = predict_start + slide_range
#         return self.train_data, self.test_data

#     def _check_start(self, date_predict_start, all_data):
#         date_predict_start = datetime.datetime.strptime(
#             date_predict_start, '%Y-%m-%d').date()
#         check = date_predict_start in all_data.index
#         while (check == False):
#             date_predict_start = date_predict_start + \
#                 relativedelta(days=+1)
#             check = date_predict_start in list(all_data.index)
#         return date_predict_start


# if __name__ == '__main__':
#     stock_name = "^GSPC"
#     date_predict_start = '2020-01-01'
#     data_range = 10
#     slide_range = 10
#     n_slide = 5
#     dataloader = LoadData(n_slide, data_range)
#     train_data, test_data =\
#         dataloader.load_and_split_data(stock_name, date_predict_start, data_range, slide_range, n_slide)
#     print(train_data)

import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import numpy as np

class LoadData:
    def __init__(self, n_slide, data_range):
        """Initializes the LoadData class.

        Args:
            n_slide (int): The number of slides.
            data_range (int): The data range.
        """
        self.train_data = np.ndarray(shape=(n_slide, data_range), dtype=float)
        self.test_data = np.ndarray(shape=(n_slide, data_range), dtype=float)
        self.date_data_start_list = []
        self.date_predict_start_list = []
        self.date_predict_end_list = []
        self.test_data_start_list = []

    def load_and_split_data(self, stock_name, date_predict_start, data_range, slide_range, n_slide):
        """Loads and splits the data.

        Args:
            stock_name (str): The stock name.
            date_predict_start (str): The start date for prediction.
            data_range (int): The data range.
            slide_range (int): The slide range.
            n_slide (int): The number of slides.

        Returns:
            numpy.ndarray, numpy.ndarray: The training and test data arrays.
        """
        all_data = yf.Ticker(stock_name).history(period='max')
        all_data.index = all_data.index.date
        # all_data['Close'].iloc[0:len(all_data['Close'])] = list(range(0, len(all_data['Close'])))
        date_predict_start = self._check_start(date_predict_start, all_data)
        predict_start = all_data.index.get_loc(date_predict_start)
        for i in range(n_slide):
            predict_end = predict_start + data_range
            data_start = predict_start - data_range
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
    data_range = 10
    slide_range = 10
    n_slide = 5
    for i in tqdm(range(0, 3000)):
        dataloader = LoadData(n_slide, data_range)
        train_data, test_data =\
            dataloader.load_and_split_data(stock_name, date_predict_start, data_range, slide_range, n_slide)
    # dataloader = LoadData(n_slide, data_range)
    # train_data, test_data =\
    #     dataloader.load_and_split_data(stock_name, date_predict_start, data_range, slide_range, n_slide)