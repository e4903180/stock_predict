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
            nonzero_indices = np.nonzero(lead_test[window])[0]
            first_nonzero_index = nonzero_indices[0]
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
