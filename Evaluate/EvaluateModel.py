import pandas as pd
import numpy as np

class Evaluate:
    def __init__(self):
        pass


    def _built_result_table(self, processed_signal, test_data_index, lead, pv, best_fit_harm=None, best_fit_error=None):
        result_table = pd.DataFrame(columns=[
            's_date', 't_date', 'lead', 'ans_date', 'pv', 'best_fit', 'error'])
        for window in range(0, processed_signal.shape[0]):
            if not type(best_fit_error) == None:
                result_table.loc[window, 'error'] = round(best_fit_error[window], 2)
            if not type(best_fit_harm) == None:
                result_table.loc[window, 'best_fit'] = best_fit_harm[window]
            result_table.loc[window, 'lead'] = lead[window]
            if pv[window] == 1:
                result_table.loc[window, 'pv'] = 'peak'
            elif pv[window] == -1:
                result_table.loc[window, 'pv'] = 'valley'
        result_table['s_date'] = test_data_index
        return result_table

    def _result_table_process(self, result_table, all_data_index, first_date):
        for window in result_table.index:
            t_date_index = np.where(all_data_index==result_table.loc[window, 's_date'])[0][0] + first_date[window]
            result_table.loc[window, 't_date'] = all_data_index[t_date_index]
            ans_date_index = t_date_index + result_table.loc[window, 'lead']
            ans_date = all_data_index[ans_date_index]
            result_table.loc[window, 'ans_date'] = ans_date

    def _compute_average_lead(self, result_table):
        final_error = round(
            sum([abs(ele) for ele in result_table['lead']]) / len(result_table['lead']), 2)
        return final_error

    def evaluate_model(self, processed_signal, all_data_index, test_data_index, best_fit_harm, best_fit_error, first_date, lead, pv):
        result_table = self._built_result_table(
            processed_signal, test_data_index, lead, pv, best_fit_harm, best_fit_error)
        self._result_table_process(result_table, all_data_index, first_date)
        final_error = self._compute_average_lead(result_table)
        return result_table, final_error
    
    