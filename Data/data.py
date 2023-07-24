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
        total_windows (int): Total number of windows for splitting data.
        columns_num (int): Number of columns to consider from the stock data.
        columns (list): List of column names to consider from the stock data.
        train_data (numpy.ndarray): Array to store training data. Shape: (total_windows, window_length, columns_num)
        train_data_index (numpy.ndarray): Array to store indices of training data. Shape: (total_windows, window_length)
        test_data (numpy.ndarray): Array to store testing data. Shape: (total_windows, window_length, columns_num)
        test_data_index (numpy.ndarray): Array to store indices of testing data. Shape: (total_windows, window_length)
    """

    def __init__(self, stock_name, window_length, slide_range, total_windows, input_num=1):
        self.stock_name = stock_name
        self.window_length = int(window_length)
        self.slide_range = int(slide_range)
        self.total_windows = int(total_windows)
        self.columns_num = input_num - 1 if input_num != 1 else 1
        self.columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        self.train_data = np.zeros(shape=(total_windows, window_length, self.columns_num), dtype=float)
        self.train_data_index = np.zeros(shape=(total_windows, window_length), dtype=object)
        self.test_data = np.zeros(shape=(total_windows, window_length, self.columns_num), dtype=float)
        self.test_data_index = np.zeros(shape=(total_windows, window_length), dtype=object)

        self.date_data_start_list = []
        self.date_predict_start_list = []
        self.date_predict_end_list = []
        self.test_data_start_list = []

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
        for i in range(self.total_windows):
            predict_end = predict_start + self.window_length
            data_start = predict_start - self.window_length
            self.train_data[i, :] = all_data[self.columns[:self.columns_num]].iloc[data_start:predict_start].values
            self.train_data_index[i, :] = all_data.iloc[data_start:predict_start].index
            self.test_data[i, :] = all_data[self.columns[:self.columns_num]].iloc[predict_start:predict_end].values
            self.test_data_index[i, :] = all_data.iloc[predict_start:predict_end].index
            predict_start = predict_start + self.slide_range
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


import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Preprocess:
    """
    Class to preprocess training and testing data.

    Attributes:
        x_length (int): Length of the input sequence.
        y_length (int): Length of the target sequence.
    """

    def __init__(self, x_length, y_length, slide=1):
        self.x_length = x_length
        self.y_length = y_length

    def preprocess_train_data(self, train_data, slide=1):
        """
        Preprocess training data.

        Args:
            train_data (numpy.ndarray): Training data. Shape: (number of windows, window length, number of features)
            self.x_length (int): Length of the input sequence.
            self.y_length (int): Length of the target sequence.
            slide (int): Sliding window step size. Default is 1.

        Returns:
            Tuple containing preprocessed input data, target data, and scaler object.
        """
        split_num = len(range(self.x_length + self.y_length, train_data.shape[1])) + 1
        train_x = np.zeros((train_data.shape[0], split_num, self.x_length, train_data.shape[-1]))
        train_y = np.zeros((train_data.shape[0], split_num, self.y_length))
        scaler = MinMaxScaler(feature_range=(0, 1))
        for window in range(train_data.shape[0]):
            for columns in range(train_data.shape[-1]):
                scaled_data_train = scaler.fit_transform(train_data[window, :, columns].reshape(-1, 1)).flatten()
                split_count = 0
                for i in range(self.x_length + self.y_length, len(scaled_data_train) + 1):
                    train_x[window, split_count, :, columns] = scaled_data_train[i - self.x_length - self.y_length:i - self.y_length]
                    if columns == 3:
                        train_y[window, split_count] = scaled_data_train[i - self.y_length:i]
                    split_count += 1
        return train_x, train_y, scaler

    def preprocess_test_data(self, scaling_data, test_data, slide=1):
        """
        Preprocess testing data.

        Args:
            scaling_data (numpy.ndarray): Data used for scaling. Shape: (number of windows, window length, number of features)
            test_data (numpy.ndarray): Testing data. Shape: (number of windows, window length, number of features)
            self.x_length (int): Length of the input sequence.
            self.y_length (int): Length of the target sequence.
            slide (int): Sliding window step size. Default is 1.

        Returns:
            Tuple containing preprocessed input data, target data, and scaler object.
        """
        split_num = len(range(self.x_length + self.y_length, test_data.shape[1], slide)) + 2
        test_x = np.zeros((test_data.shape[0], split_num, self.x_length, test_data.shape[-1]))
        test_y = np.zeros((test_data.shape[0], split_num, self.y_length))
        scaler = MinMaxScaler(feature_range=(0, 1))
        for window in range(test_data.shape[0]):
            for columns in range(test_data.shape[-1]):
                scaled_data_train = scaler.fit_transform(scaling_data[window, :, columns].reshape(-1, 1)).flatten()
                scaled_data_test = scaler.fit_transform(test_data[window, :, columns].reshape(-1, 1)).flatten()
                split_count = 0
                test_x[window, split_count, :, columns] = scaled_data_train[-self.x_length:]
                if columns == 3:
                    test_y[window, split_count] = scaled_data_test[:self.y_length]
                split_count += 1
                for i in range(self.x_length + self.y_length, len(scaled_data_test) + slide, slide):
                    test_x[window, split_count, :, columns] = scaled_data_test[i - self.x_length - self.y_length:i - self.y_length]
                    if columns == 3:
                        test_y[window, split_count] = scaled_data_test[i - self.y_length:i]
                    split_count += 1
        return test_x, test_y, scaler


import numpy as np
import math

class Postprocess:
    """A class for postprocessing signals."""
    def __init__(self):
        pass

    def postprocess_lstm(self, predicted_y, scaler):
        """Postprocess the predicted LSTM signal.

        Args:
            predicted_y: numpy.ndarray
                Predicted signal obtained from predictions.
                shape = (number of windows, number of split y, length of y)
            scaler: object
                Scaler object used for inverse transformation.

        Returns:
            processed_lstm_signal: numpy.ndarray
                Processed signal flattened from predicted_y.
                shape = (number of windows, 1, window_length)
        """
        inversed_signal = self._inverse_transfer(predicted_y, scaler)
        processed_lstm_signal = self._flatten(inversed_signal)
        return processed_lstm_signal
    
    def _inverse_transfer(self, predicted_y, scaler):
        """
        Reverse the transformation of the predicted signal.

        Args:
            predicted_y: numpy.ndarray
                Predicted signal obtained from predictions.
                shape = (number of windows, number of split y, length of y)

        Returns:
            processed_signal: numpy.ndarray
                Processed signal flatten from predicted_y.
                shape = (number of windows, 1, window_length)
        """

        inversed_signal = np.zeros_like(predicted_y)
        for window in range(0, predicted_y.shape[0]):
            inversed_signal[window] = scaler.inverse_transform(predicted_y[window])
        return inversed_signal
    
    def _flatten(self, inversed_signal):
        """
        Flatten the predicted signal.

        Args:
            inversed_signal: numpy.ndarray
                Predicted signal obtained from predictions.
                shape = (number of windows, number of split y, length of y)

        Returns:
            processed_signal: numpy.ndarray
                Processed signal flatten from predicted_y.
                shape = (number of windows, 1, window_length)
        """

        flatten_signal = np.ndarray([inversed_signal.shape[0], inversed_signal.shape[1]*inversed_signal.shape[2]])
        for window in range(0, inversed_signal.shape[0]):
            flatten_signal[window] = inversed_signal[window].flatten()
        return flatten_signal
    
    def postprocess_fft(self, mixed_train_harm, mixed_test_harm, pv_range, pv_train_data, fit_method):
        """Postprocess the FFT signal.

        Args:
            mixed_train_harm: numpy.ndarray
                Mixed harmonics of the training data.
                shape: (number of windows, number of mixed harmonics, window_length)
            mixed_test_harm: numpy.ndarray
                Mixed harmonics of the test data.
                shape: (number of windows, number of mixed harmonics, window_length)
            pv_range: int
                The range within which peaks and valleys are detected.
            pv_train_data: numpy.ndarray
                Training data used for finding peaks and valleys.
                shape: (number of windows, window_length)
            fit_method: str
                The method used for fitting the signal.

        Returns:
            processed_signal: numpy.ndarray
                Processed signal.
                shape: (number of windows, window_length)
            best_fit_harm: numpy.ndarray
                Indices of the best fitting harmonics for each window.
                shape: (number of windows)
            best_fit_error: numpy.ndarray
                Error values of the best fitting harmonics for each window.
                shape: (number of windows)
        """
        pv_mixed_train_harm = self.find_data_pv(mixed_train_harm, pv_range)
        lead_mixed_train_harm = self.find_fft_lead(pv_train_data, pv_mixed_train_harm)
        errors = self._get_fit_error(lead_mixed_train_harm, fit_method)
        best_fit_harm, best_fit_error = self._get_best_fit(errors)
        processed_signal = self._pick_best_fit_harm(mixed_test_harm, best_fit_harm)
        return processed_signal, best_fit_harm, best_fit_error

    def find_data_pv(self, data, pv_range):
        '''
        Find peaks and valleys of the data, excluding the first and last data points.

        Args:
            data: numpy.ndarray
                The input data array.
                    fft_processed_signal: numpy.ndarray 
                        The processed signal obtained by mixing the selected harmonics.
                        shape: (number of windows, number of mixed harmonics, window_length)
                        
                    lstm_y: numpy.ndarray
                        Processed signal obtained from predictions.
                        shape = (number of windows, number of split y, length of y)
                        flatten -> (number of windows, window_length)

            pv_range: int
                The range within which peaks and valleys are detected.

        Returns:
            pv: numpy.ndarray
                An array of the same shape as `data`, where peaks are represented by 1,
                valleys are represented by -1, and other points are represented by 0.
                shape: (number of windows, window_length)
        Raises:
            None
        '''
        pv = np.zeros_like(data)
        if len(pv.shape) == 3:
            for window in range(0, data.shape[0]):
                for harmonics in range(0, data.shape[1]):
                    pv[window, harmonics] = self._find_pv(data[window, harmonics], pv_range)
        elif len(pv.shape) == 2:
            for window in range(0, data.shape[0]):
                pv[window] = self._find_pv(data[window], pv_range)
        return pv
    
    def _find_pv(self, data, pv_range):
        """Find peaks and valleys in the data.

        Args:
            data: numpy.ndarray
                The input data array.
            pv_range: int
                The range within which peaks and valleys are detected.

        Returns:
            pv: numpy.ndarray
                Array representing peaks (1), valleys (-1), and other points (0).

        Raises:
            None
        """
        pv = np.zeros_like(data)
        for l in range(1, data.shape[0]): # Ignore pv on the first data point
            if l < pv_range:
                if data[l] == data[:l+pv_range+1].max():
                    pv[l] = 1
                if data[l] == data[:l+pv_range+1].min():
                    pv[l] = -1   
            else:
                if data[l] == data[l-pv_range:l+pv_range+1].max():
                    pv[l] = 1   
                if data[l] == data[l-pv_range:l+pv_range+1].min():
                    pv[l] = -1
        return pv
    
    def _find_peak_lead(self, element, pv_data, pv_signal):
        """Find the lead for a peak element.

        Args:
            element: tuple
                Tuple containing the index and value of the element.
            pv_train_data: numpy.ndarray
                Data array used for finding peaks and valleys.
                shape: (window_length)
            pv_signal: numpy.ndarray
                Signal array containing peaks (1), valleys (-1), and others (0).
                shape: (window_length)

        Returns:
            lead: int
                Lead value for the peak element.

        Raises:
            None
        """
        front = 'NULL'
        back = 'NULL'
        lead = None
        forword = list(range(0, len(pv_signal)-element[0]))
        backword = list(range(0, element[0]+1))
        for i in forword:
            if pv_data[element[0]+i] == 1:
                front = i
                break 
        for i in backword:
            if pv_data[element[0]-i] == 1:
                back = -i
                break
        if front == 'NULL' and back == 'NULL':
            print('no peaks or valleys lead in data')
        elif front != 'NULL' and back == 'NULL':
            lead = front
        elif front == 'NULL' and back != 'NULL':
            lead = back
        elif front <= np.absolute(back):
            lead = front
        elif front > np.absolute(back):
            lead = back
        return lead

    def _find_valley_lead(self, element, pv_train_data, pv_signal):
        """Find the lead for a valley element.

        Args:
            element: tuple
                Tuple containing the index and value of the element.
            pv_train_data: numpy.ndarray
                Data array used for finding peaks and valleys.
                shape: (window_length)
            pv_signal: numpy.ndarray
                Signal array containing peaks (1), valleys (-1), and others (0).
                shape: (window_length)

        Returns:
            lead: int
                Lead value for the valley element.

        Raises:
            None
        """
        front = 'NULL'
        back = 'NULL'
        lead = None
        forword = list(range(0, len(pv_signal)-element[0]))
        backword = list(range(0, element[0]+1))
        for i in forword:
            if pv_train_data[element[0]+i] == -1:
                front = i
                break 
        for i in backword:
            if pv_train_data[element[0]-i] == -1:
                back = -i
                break
        if front == 'NULL' and back == 'NULL':
            print('no peaks or valleys lead in data')
        elif front != 'NULL' and back == 'NULL':
            lead = front
        elif front == 'NULL' and back != 'NULL':
            lead = back
        elif front <= np.absolute(back):
            lead = front
        elif front > np.absolute(back):
            lead = back
        return lead
    
    def find_lead(self, pv_data, pv_signal):
        """Find the lead values for peak and valley elements.

        Args:
            pv_data: numpy.ndarray
                Data array used for finding peaks and valleys.
                shape: (number of windows, window_length)
            pv_signal: numpy.ndarray
                Signal array containing peaks (1), valleys (-1), and others (0).
                shape: (number of windows, window_length)

        Returns:
            lead: numpy.ndarray
                Array containing the lead values for each element.
                shape: (number of windows, window_length)

        Raises:
            None
        """
        lead = np.ndarray([pv_signal.shape[0], pv_signal.shape[1]], dtype='object')
        for window in range(0, pv_signal.shape[0]):
            for element in enumerate(pv_signal[window]):
                if element[1] == 1:
                    lead[window, element[0]] = self._find_peak_lead(element, pv_data[window], pv_signal[window])
                elif element[1] == -1:
                    lead[window, element[0]] = self._find_valley_lead(element, pv_data[window], pv_signal[window])
                else:
                    lead[window, element[0]] = None
        return lead

    def find_fft_lead(self, pv_data, pv_fft_signal):
        """Find the lead values for peak and valley elements in the FFT signal.

        Args:
            pv_data: numpy.ndarray
                Data array used for finding peaks and valleys.
                shape: (number of windows, window_length)
            pv_signal: numpy.ndarray
                Signal array containing peaks (1), valleys (-1), and others (0).
                shape: (number of windows, window_length)

        Returns:
            lead: numpy.ndarray
                Array containing the lead values for each element.
                shape: (number of windows, window_length)

        Raises:
            None
        """
        lead = np.ndarray([pv_fft_signal.shape[0], pv_fft_signal.shape[1], pv_fft_signal.shape[2]], dtype='object')
        for window in range(0, pv_fft_signal.shape[0]):
            for harm in range(0, pv_fft_signal.shape[1]):
                for element in enumerate(pv_fft_signal[window, harm]):
                    if element[1] == 1:
                        lead[window, harm, element[0]] = self._find_peak_lead(element, pv_data[window], pv_fft_signal[window, harm])
                    elif element[1] == -1:
                        lead[window, harm, element[0]] = self._find_valley_lead(element, pv_data[window], pv_fft_signal[window, harm])
                    else:
                        lead[window, harm, element[0]] = None
        return lead
    
    def _compute_error_maen(self, lead):
        """Compute the mean error value.

        Args:
            lead: numpy.ndarray
                Array containing the lead values.

        Returns:
            error: float
                Mean error value.

        Raises:
            None
        """
        error = 0
        for num in lead:
            if num is not None:
                error += num
        return error/lead.shape[0]

    def _compute_error_abs(self, lead):
        """Compute the absolute error value.

        Args:
            lead: numpy.ndarray
                Array containing the lead values.

        Returns:
            error: float
                Absolute error value.

        Raises:
            None
        """
        error = 0
        for num in lead:
            if num is not None:
                error += np.abs(num)
        return error/lead.shape[0]

    def _compute_error_rmse(self, lead):
        """Compute the root mean square error value.

        Args:
            lead: numpy.ndarray
                Array containing the lead values.

        Returns:
            error: float
                Root mean square error value.

        Raises:
            None
        """
        error = 0
        for num in lead:
            if num is not None:
                # error += np.abs(num)
                mse = np.square(num).mean()
                rmse = math.sqrt(mse)
                error += rmse
        return error/lead.shape[0]

    def _get_fit_error(self, lead_mixed_train_harm, fit_method):
        """Compute the error values for fitting.

        Args:
            lead_mixed_train_harm: numpy.ndarray
                Array containing the lead values for each element.
                shape: (number of windows, number of harmonics)
            fit_method: str
                The method used for fitting the signal.

        Returns:
            errors: numpy.ndarray
                Array containing the error values.
                shape: (number of windows, number of harmonics)

        Raises:
            None
        """
        errors = np.ndarray([lead_mixed_train_harm.shape[0], lead_mixed_train_harm.shape[1]])
        error = int()
        for i in range(0, lead_mixed_train_harm.shape[0]):
            for j in range(0, lead_mixed_train_harm.shape[1]):
                if fit_method == 'mean':
                    error = self._compute_error_maen(lead_mixed_train_harm[i, j])
                elif fit_method == 'abs':
                    error = self._compute_error_abs(lead_mixed_train_harm[i, j])
                elif fit_method == 'rmse':
                    error = self._compute_error_rmse(lead_mixed_train_harm[i, j])
                else :
                    print('worng fit_method')
                errors[i, j] = error
        return errors

    def _get_best_fit(self, errors):
        """Get the best fitting harmonics and their error values.

        Args:
            errors: numpy.ndarray
                Array containing the error values.
                shape: (number of windows, number of harmonics)

        Returns:
            best_fit_harm: numpy.ndarray
                Array containing the indices of the best fitting harmonics.
                shape: (number of windows)
            best_fit_error: numpy.ndarray
                Array containing the error values of the best fitting harmonics.
                shape: (number of windows)

        Raises:
            None
        """
        best_fit_harm = np.ndarray(errors.shape[0])
        best_fit_error = np.ndarray(errors.shape[0])
        for window in range(errors.shape[0]):
            best_fit_harm[window] = np.argmin(errors[window])
            best_fit_error[window] = errors[window, int(best_fit_harm[window])]
        return best_fit_harm, best_fit_error

    def _pick_best_fit_harm(self, mixed_test_harm, best_fit_harm):
        """Pick the best fitting harmonics.

        Args:
            mixed_test_harm: numpy.ndarray
                Mixed harmonics of the test data.
                shape: (number of windows, number of mixed harmonics, window_length)
            best_fit_harm: numpy.ndarray
                Array containing the indices of the best fitting harmonics.
                shape: (number of windows)

        Returns:
            processed_signal: numpy.ndarray
                Processed signal.
                shape: (number of windows, window_length)

        Raises:
            None
        """
        processed_signal = np.ndarray([mixed_test_harm.shape[0], mixed_test_harm.shape[2]])
        for window in range(0, mixed_test_harm.shape[0]):
            processed_signal[window] = mixed_test_harm[window, int(best_fit_harm[window])]
        return processed_signal

    def get_first_lead(self, pv_signal, lead_test):
        """Get the first lead values for each window.

        Args:
            pv_signal: numpy.ndarray
                Signal array containing peaks (1), valleys (-1), and others (0).
                shape: (number of windows, window_length)
            lead_test: numpy.ndarray
                Array containing the lead values for each element.
                shape: (number of windows, window_length)

        Returns:
            first_date: numpy.ndarray
                Array containing the index of the first non-zero element in each window.
                shape: (number of windows)
            lead: numpy.ndarray
                Array containing the lead values for each window.
                shape: (number of windows)
            pv: numpy.ndarray
                Array containing the peak/valley values for each window.
                shape: (number of windows)

        Raises:
            None
        """
        first_date = np.zeros(pv_signal.shape[0], dtype=object)
        lead = np.zeros(pv_signal.shape[0], dtype=object)
        pv = np.zeros(pv_signal.shape[0], dtype=object)
        for window in range(0, pv_signal.shape[0]):
            for i in range(0, pv_signal.shape[1]):
                if lead_test[window, i] != None:
                    first_date[window] = i
                    lead[window] = lead_test[window, i]
                    pv[window] = pv_signal[window, i]
                    break
            if pv[window] == 0:
                first_date[window] = None
                lead[window] = None
                pv[window] = None
                print(f'no peaks or valleys lead in data{window}')
        return first_date, lead, pv
