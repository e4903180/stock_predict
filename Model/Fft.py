import sys
import os
import numpy as np

module_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from data.loadData import LoadData
from data.preprocess import Preprocess

class Fft():
    def __init__(self):
        pass

    def fft(self, train_data, window_length, n_harm_lower_limit, n_harm_upper_limit):
        harmonics = self._extract(train_data, window_length)
        processed_signal = self._mix(harmonics, n_harm_lower_limit, n_harm_upper_limit)
        return processed_signal, harmonics

    def _extract(self, train_data, window_length):
        '''Todo correct name of parm, order of parm.
        Convert input data to harmonics using FFT (Fast Fourier Transform).

        Args:
            train_data: numpy.ndarray
                The input training data.
            window_length: int
                The range of data used for harmonic extraction.

        Returns:
            harmonics: numpy.ndarray
                An array of harmonics extracted from the input data.
                shape: (number of windows, number of positive frequencies, window_length)

        Raises:
            None
        '''
        f_positive_num = int((train_data.shape[1]/2)-1)
        harmonics = np.ndarray((train_data.shape[0], f_positive_num, train_data.shape[1]))
        # (number of windows, number of positive frequencies, window_length)
        for window in range(0, train_data.shape[0]):
            # get data_stock's infomation
            data = train_data[window]
            array_data = np.array(data)
            n_data = array_data.size
            time_data = np.arange(0, n_data)

            # detrend data
            # find linear trend in data
            Polynomial = np.polyfit(time_data, array_data, 1)
            data_notrend = array_data - Polynomial[0] * time_data    # detrended x

            # fft process
            data_freqdom = np.fft.fft(data_notrend, n=n_data)
            frequence = np.fft.fftfreq(n=n_data, d=1)
            f_positive = frequence[np.where(frequence > 0)]
            data_freqdom_positive = data_freqdom[np.where(frequence > 0)]
            # sort indexes
            indexes = list(range(f_positive.size))      # frequencies
            # sort method 1
            # indexes.sort(key = lambda window: np.absolute(frequence[window]))     # sort indexes by frequency, lower -> higher
            # sort method 2 :
            # sort indexes by amplitudes, lower -> higher
            indexes.sort(key=lambda window: np.absolute(data_freqdom[window]))
            indexes.reverse()       # sort indexes by amplitudes, higher -> lower

            # get data_all_time'size
            time_transfer = np.arange(window_length, window_length*2)
            count=0
            # get harmonics
            for j in indexes:
                ampli = np.absolute(
                    data_freqdom_positive[j]) / n_data     # amplitude
                phase = np.angle(data_freqdom_positive[j])      # phase
                harmonics[window, j] = ampli * \
                    np.cos(2 * np.pi * f_positive[j] * time_transfer + phase)
                count+=1
                # print(j, count)
        return harmonics

    def _mix(self, harmonics, n_harm_lower_limit, n_harm_upper_limit):
        '''
        Mix a specified range of harmonics to generate a processed signal.

        Args:
            harmonics: numpy.ndarray
                The harmonics extracted from the input data.
            n_harm_lower_limit: int
                The lower limit of the number of harmonics to be mixed.
            n_harm_upper_limit: int
                The upper limit of the number of harmonics to be mixed.

        Returns:
            processed_signal: numpy.ndarray 
                The processed signal obtained by mixing the selected harmonics.
                shape: (number of windows, number of mixed harmonics, window_length)

        Raises:
            None
        '''
        mixed_harmonic_num = n_harm_upper_limit - n_harm_upper_limit + 1 
        processed_signal = np.ndarray((harmonics.shape[0], (n_harm_upper_limit-n_harm_lower_limit+1), harmonics.shape[2]))
        # (number of windows, number of mixed harmonics, window_length*2)
        for window in range(processed_signal.shape[0]):
            for n_harm in range(processed_signal.shape[1]):
                mixed_harmonic = np.zeros(processed_signal.shape[2])
                for j in range(mixed_harmonic_num):
                    mixed_harmonic += harmonics[window][j]
                processed_signal[window][n_harm] = mixed_harmonic
        return processed_signal


if __name__ == '__main__':
    stock_name = "^GSPC"
    date_predict_start = '2020-01-01'
    window_length = 30
    slide_range = 40
    total_windows = 3
    slide = 5
    n_harm_lower_limit = 1
    n_harm_upper_limit = 30
    dataloader = LoadData(total_windows, window_length)
    train_data, test_data =\
        dataloader.load_and_split_data(stock_name, date_predict_start, window_length, slide_range, total_windows)
    x_length = 3
    y_length = 5
    # preprocesser = Preprocess()
    # x_train, y_train =\
    #     preprocesser.preprocess_data(train_data, x_length, y_length)
    model = Fft()
    processed_signal, harmonics = model.fft(train_data, window_length, n_harm_lower_limit, n_harm_upper_limit)

    # harmonics = model.data_to_harmonics_function(train_data, window_length)
    # processed_signal = model.mix_harmonics_function(harmonics, n_harm_lower_limit, n_harm_upper_limit)
    # print(processed_signal)
    # print(harmonics.shape)
    # print(processed_signal.shape)



