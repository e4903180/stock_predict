import numpy as np
from sklearn.preprocessing import MinMaxScaler

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from data.loadData import LoadData

class Preprocess:
    def __init__(self):
        pass

    def preprocess_train_data(self, train_data, x_length, y_length, slide=1):
        split_num = len(range(x_length+y_length, train_data.shape[1]))+1
        train_x = np.ndarray((train_data.shape[0], split_num, x_length, 1)) 
        train_y = np.ndarray((train_data.shape[0], split_num, y_length, 1)) 
        scaler = MinMaxScaler(feature_range=(0,1))
        for window in range(0, train_data.shape[0]):
            scaled_data_train = scaler.fit_transform(train_data[window].reshape(-1,1))
            split_count = 0
            for i in range(x_length+y_length, len(scaled_data_train)+1):
                train_x[window, split_count] = scaled_data_train[i-x_length-y_length:i-y_length]
                train_y[window, split_count] = scaled_data_train[i-y_length:i]
                split_count += 1
        return train_x, train_y, scaler
    
    def preprocess_test_data(self, train_data, test_data, x_length, y_length, slide=1):
            split_num = len(range(x_length+y_length, test_data.shape[1], slide))+2
            test_x = np.ndarray((test_data.shape[0], split_num, x_length, 1)) 
            test_y = np.ndarray((test_data.shape[0], split_num, y_length, 1)) 
            scaler = MinMaxScaler(feature_range=(0,1))
            for window in range(0, test_data.shape[0]):
                scaled_data_train = scaler.fit_transform(train_data[window].reshape(-1,1))
                scaled_data_test = scaler.fit_transform(test_data[window].reshape(-1,1))
                split_count = 0
                test_x[window, split_count] = scaled_data_train[-x_length:]
                test_y[window, split_count] = scaled_data_test[:y_length]
                split_count += 1
                for i in range(x_length+y_length, len(scaled_data_test)+slide, slide):
                    test_x[window, split_count] = scaled_data_test[i-x_length-y_length:i-y_length]
                    test_y[window, split_count] = scaled_data_test[i-y_length:i]
                    split_count += 1
            return test_x, test_y, scaler
