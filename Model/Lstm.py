import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from tqdm import tqdm
import sys
import os
module_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from data.loadData import LoadData
from data.preprocess import Preprocess

class Lstm:
    def __init__(self):
        pass
    
    def lstm(self, x_train, y_train, x_test, y_test, train_data, test_data, y_length):
        """
        Trains and predicts using LSTM model.

        Args:
            x_train (array): Training input data.
            y_train (array): Training target data.
            x_test (array): Test input data.
            y_test (array): Test target data.
            test_data (array): Data for prediction.
            train_data (array): Training data.
            y_length (int): Length of the target data.

        Returns:
            array: Processed signal obtained from predictions.
        """
        model = self._build(x_train, y_length)
        history = self._train(train_data, model, x_train, y_train)
        predicted_y = self._predict(x_test, y_test, test_data, model)
        # processed_signal = self._flatten(predicted_y)
        return predicted_y

    def _construct(self, y_length, input_shape):
        """
        Constructs an LSTM model.

        Args:
            y_length (int): Length of the target data.
            input_shape (tuple): Shape of the input data.

        Returns:
            Sequential: Constructed LSTM model.
        """
        model = Sequential()
        model.add(LSTM(units = 256, return_sequences = True, input_shape = input_shape))
        model.add(Dropout(0.4))
        model.add(LSTM(units = 128, return_sequences = True))
        model.add(Dropout(0.4))
        model.add(LSTM(units = 64))
        model.add(Dropout(0.4))
        model.add(Dense(units=y_length))
        return model
    
    def _build(self, x_train, y_length):
        """
        Builds a list of LSTM models.

        Args:
            x_train (array): Training input data.
            y_length (int): Length of the target data.

        Returns:
            list: List of built LSTM models.
        """
        model = list(range(0, x_train.shape[0]))
        input_shape = (x_train.shape[2],1)
        for i in range(0, x_train.shape[0]):
            model[i] = self._construct(y_length, input_shape)
            model[i].compile(optimizer='adam', 
                        loss='mean_squared_error')
            # model[i].summary()
        return model
    
    def _train(self, train_data, model, x_train, y_train):
        """
        Trains the LSTM models.

        Args:
            train_data (array): Training data.
            model (list): List of LSTM models.
            x_train (array): Training input data.
            y_train (array): Training target data.

        Returns:
            list: List of training histories.
        """
        history = list(range(0, train_data.shape[0]))
        for i in tqdm(range(0, train_data.shape[0])):
            earlystopper = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 2)
            history[i] = model[i].fit(x_train[i], y_train[i], epochs=25, batch_size=32, 
                                callbacks = [earlystopper], validation_split = 0.2, shuffle=True)
        return history
    
    def _predict(self, x_test, y_test, test_data, model):
        """
        Generates predictions using the trained models.

        Args:
            x_test (array): Test input data.
            y_test (array): Test target data.
            test_data (array): Data for prediction.
            model (list): List of LSTM models.

        Returns:
            predicted_y: numpy.ndarray
                Processed signal obtained from predictions.
                shape = (number of windows, number of split y, length of y)
        """
        predicted_y = np.ndarray([y_test.shape[0], y_test.shape[1], y_test.shape[2]])
        for i in tqdm(range(0, test_data.shape[0])):
            predicted_prices = model[i].predict(x_test[i])
            predicted_y[i] = predicted_prices
        return predicted_y
    
    # def _flatten(self, predicted_y):
    #     """
    #     flatten predictions.

    #     Args:
    #         predicted_y: numpy.ndarray
    #             Predicted signal obtained from predictions.
    #             shape = (number of windows, number of split y, length of y)

    #     Returns:
    #         processed_signal: numpy.ndarray
    #             Processed signal flatten from predicted_y.
    #             shape = (number of windows, 1, window_length)
    #     """
    #     processed_signal = np.ndarray([predicted_y.shape[0], 1, predicted_y.shape[1]*predicted_y.shape[2]])
    #     for window in range(0, predicted_y.shape[0]):
    #         processed_signal[window][0] = predicted_y[window].flatten()
    #     return processed_signal
    
if __name__ == '__main__':
    stock_name = "^GSPC"
    date_predict_start = '2020-01-01'
    window_length = 30
    slide_range = 40
    total_windows = 3
    slide = 5
    dataloader = LoadData(total_windows, window_length)
    train_data, test_data =\
        dataloader.load_and_split_data(stock_name, date_predict_start, window_length, slide_range, total_windows)
    x_length = 5
    y_length = 5
    preprocesser = Preprocess()
    x_train, y_train, scaler_train =\
        preprocesser.preprocess_train_data(train_data, x_length, y_length)
    x_test, y_test, scaler_test =\
        preprocesser.preprocess_test_data(train_data, test_data, x_length, y_length, slide)
    lstm = Lstm()
    processed_signal = lstm.lstm(x_train, y_train, x_test, y_test, test_data, train_data, y_length)
    # model = lstm._build(x_train, y_length)
    # history = lstm._train(train_data, model, x_train, y_train)
    # processed_signal = lstm._predict(x_test, y_test, test_data, model)
    print(processed_signal)
    print(processed_signal.shape)