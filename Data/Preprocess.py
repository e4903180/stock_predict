import numpy as np
from sklearn.preprocessing import MinMaxScaler

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
if module_path not in sys.path:
    sys.path.append(module_path)

    
from Data.LoadData import LoadData

from line_profiler import LineProfiler

class Preprocess:
    def __init__(self):
        pass

    @profile
    def preprocess_data(self, data, x_length, y_length, slide=1):
        '''
        Split and standardize data.
        Preprocess the data for LSTM model.

        Args:
            data: numpy.ndarray
                The input data array.
            x_length: int
                The length of the input sequence.
            y_length: int
                The length of the output sequence.
            slide: int, optional
                The sliding window step size. Default is 1.

        Returns:
            x: numpy.ndarray
                The preprocessed input training data.
                shape: (number of windows, number of split x, length of x, 1)
            y: numpy.ndarray
                The preprocessed output training data.
                shape: (number of windows, number of split y, length of y, 1)

        Raises:
            None
        '''

        split_num = len(range(x_length+y_length, data.shape[1], slide))
        x = np.ndarray((data.shape[0], split_num, x_length, 1)) 
        y = np.ndarray((data.shape[0], split_num, y_length, 1)) 
        scaler = MinMaxScaler(feature_range=(-1,1))
        for window in range(0, data.shape[0]):
            scaled_data = scaler.fit_transform(data[window].reshape(-1,1))
            split_count = 0
            for i in range(x_length+y_length, len(scaled_data), slide):
                x[window, split_count] = scaled_data[i-x_length-y_length:i-y_length]
                y[window, split_count] = scaled_data[i-y_length:i]
                split_count += 1
        return x, y


if __name__ == '__main__':
    stock_name = "^GSPC"
    date_predict_start = '2020-01-01'
    window_length = 10
    slide_range = 10
    total_windows = 5
    dataloader = LoadData(total_windows, window_length)
    train_data, test_data =\
        dataloader.load_and_split_data(stock_name, date_predict_start, window_length, slide_range, total_windows)
    
    x_length = 2
    y_length = 2
    preprocesser = Preprocess()
    x_train, y_train =\
        preprocesser.preprocess_data(train_data, x_length, y_length, 2)
    # print(x_train.shape)

    x_test, y_test =\
        preprocesser.preprocess_data(test_data, x_length, y_length, 2)
    # print(x_test.shape)
    