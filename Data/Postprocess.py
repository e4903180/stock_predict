from tqdm import tqdm
import numpy as np

class Postprocessing:
    def __init__(self):
        pass

    def find_data_pv_function(self, data, pv_range):
        '''
        Find peaks and valleys of the data, excluding the first and last data points.

        Args:
            data: numpy.ndarray
                The input data array.
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
        for col in range(0, data.shape[0]):
            for row in range(1, data.shape[1]-1):
                if row < pv_range:
                    if data[col, row] == data[col, :row+pv_range+1].max():
                        pv[col, row] = 1
                    if data[col, row] == data[col, :row+pv_range+1].min():
                        pv[col, row] = -1   
                else:
                    if data[col, row] == data[col, row-pv_range:row+pv_range+1].max():
                        pv[col, row] = 1   
                    if data[col, row] == data[col, row-pv_range:row+pv_range+1].min():
                        pv[col, row] = -1
        return pv

    def find_signal_lead_train_function(self, data, processed_signal):
        for d in data:
            for p in processed_signal[d]:
                # processed_signal[d][p]['pv'] = pd.Series(dtype='str')
                processing_signal = processed_signal[d][p].head(len(data[d]))
                p_data = pd.DataFrame(
                    {'peaks': data[d]['peaks'], 'count': range(len(data[d]))})
                p_data = p_data.drop(p_data[p_data['peaks'].isna()].index)
                p_data_count = list(p_data['count'])
                p_signal = pd.DataFrame(
                    {'peaks': processing_signal['peaks'], 'count': range(len(processing_signal))})
                p_signal = p_signal.drop(p_signal[p_signal['peaks'].isna()].index)
                p_signal_list = list(p_signal['count'])
                p_lead = []
                for i in range(0, len(p_signal_list)):
                    temp = []
                    temp_abs = []
                    temp_2 = []
                    for j in range(0, len(p_data_count)):
                        temp.append((p_data_count[j] - p_signal_list[i]))
                        temp_abs.append(abs(p_data_count[j] - p_signal_list[i]))
                    for k in range(0, len(temp_abs)):
                        if temp_abs[k] == min(temp_abs):
                            temp_2 = temp[k]
                    p_lead.append(temp_2)
                p_signal['lead'] = p_lead

                v_data = pd.DataFrame(
                    {'valleys': data[d]['valleys'], 'count': range(len(data[d]))})
                v_data = v_data.drop(v_data[v_data['valleys'].isna()].index)
                v_data_count = list(v_data['count'])
                v_signal = pd.DataFrame(
                    {'valleys': processing_signal['valleys'], 'count': range(len(processing_signal))})
                v_signal = v_signal.drop(
                    v_signal[v_signal['valleys'].isna()].index)
                v_signal_list = list(v_signal['count'])
                v_lead = []
                for i in range(0, len(v_signal_list)):
                    temp = []
                    temp_abs = []
                    temp_2 = []
                    for j in range(0, len(v_data_count)):
                        temp.append((v_data_count[j] - v_signal_list[i]))
                        temp_abs.append(abs(v_data_count[j] - v_signal_list[i]))
                    for k in range(0, len(temp_abs)):
                        if temp_abs[k] == min(temp_abs):
                            temp_2 = temp[k]
                    v_lead.append(temp_2)
                v_signal['lead'] = v_lead

                processed_signal[d][p]['lead'] = pd.Series(dtype='float64')
                processed_signal[d][p]['lead'].loc[p_signal['lead'].index] = p_signal['lead']
                processed_signal[d][p]['lead'].loc[v_signal['lead'].index] = v_signal['lead']

    def get_first_lead_function(self, processed_signal, best_fit_harm):
        first_date = {}
        lead = {}
        pv = {}
        for i in processed_signal:
            harm = best_fit_harm[i]
            temp = processed_signal[i][harm].loc[list(
                processed_signal[i][harm]['lead'].dropna().index)[0]]
            first_date[i] = list(processed_signal[i][harm]
                                ['lead'].dropna().index)[0]
            lead[i] = temp['lead']
            pv[i] = temp['pv']
        return first_date, lead, pv

    def get_fit_error_function(self, processed_signal, fit_method):
        errors = {}
        error = []
        for i in processed_signal:
            errors[i] = {}
            for j in processed_signal[i]:
                signal_dropna = processed_signal[i][j].drop(
                    processed_signal[i][j][processed_signal[i][j]['lead'].isna()].index)
                if fit_method == 'mean':
                    error = signal_dropna['lead'].mean()
                elif fit_method == 'abs':
                    error = abs(signal_dropna['lead']).mean()
                elif fit_method == 'rmse':
                    mse = np.square(np.subtract(np.zeros_like(
                        signal_dropna['lead']), signal_dropna['lead'])).mean()
                    rmse = math.sqrt(mse)
                    error = rmse
                else :
                    print('worng fit_method')
                errors[i][j] = error
        return errors
    
    def get_best_fit_harm_function(self, processed_signal, errors):
        best_error = {}
        best_fit_harm = {}
        for i in processed_signal:
            best_error[i] = pd.Series(errors[i]).abs().min()
            best_fit_harm[i] = pd.Series(errors[i]).abs().idxmin()
        return best_fit_harm, best_error



