import numpy as np
import pylab as pl
from numpy import fft
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
import math
import matplotlib.dates as mdates
from tqdm import tqdm
import matplotlib.patches as patches
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

class DataProcess:
    '''
    '''

    def __init__(self):
        '''
        Args :
            
        '''
        pass

    def load_data(self, stock_name="^GSPC", date_predict_start='2021-01-01', data_range=250, slide_range=5, n_slide=5):
        '''從yfinance下載資料,切分成train_data, test_data

            Args :
                stock_name : (str) 股票名
                date_predict_start : (str) 預測開始日期
                data_range : (int) slide window長度
                slide_range : (int) slide window間隔
                n_slide : (int) slide window數目

            Return :
                train_data : (dict) 包含train data的dataframe, train data為自date_predict_start前data_range個points
                test_data : (dict) 包含test data的dataframe, test data為自date_predict_start後data_range個points
                all_data : (dataframe) 此stock所有data
                test_data_start_list : (list) 紀錄每個test_data的開始日期
        '''
        train_data = {}
        test_data = {}
        date_predict_start = datetime.datetime.strptime(
            date_predict_start, '%Y-%m-%d').date()
        date_data_start_list = []
        date_predict_start_list = []
        date_predict_end_list = []
        test_data_start_list = []
        all_data = yf.Ticker(stock_name).history(period='max')
        all_data.index = all_data.index.date
        check = date_predict_start in all_data.index
        while (check == False):
            date_predict_start = date_predict_start + \
                relativedelta(days=+1)
            check = date_predict_start in list(all_data.index)
        predict_start = all_data.index.get_loc(date_predict_start)
        for i in range(n_slide):
            predict_end = predict_start + data_range
            date_predict_end = all_data.iloc[predict_end].name
            data_start = predict_start - data_range
            date_data_start = all_data.iloc[data_start].name
            train_data['window_' + str(i)] = all_data.iloc[data_start:predict_start]
            test_data['window_' + str(i)] = all_data.iloc[predict_start:predict_end]
            date_data_start_list.append(date_data_start)
            date_predict_start_list.append(date_predict_start)
            date_predict_end_list.append(date_predict_end)
            test_data_start_list.append(test_data['data_' + str(i)].index[0])
            data_start = data_start + slide_range
            predict_start = predict_start + slide_range
            train_data['window_' + str(i)] = train_data['window_' +
                                                    str(i)].reset_index(drop=False)
            test_data['window_' + str(i)] = test_data['window_' +
                                                    str(i)].reset_index(drop=False)

        return train_data, test_data, all_data, test_data_start_list


