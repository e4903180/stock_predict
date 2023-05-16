import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import numpy as np

class LoadData:

    def __init__(self) -> None:
        self.train_data = np.ndarray([])
        self.test_data = np.ndarray([])
        self.date_data_start_list = []
        self.date_predict_start_list = []
        self.date_predict_end_list = []
        self.test_data_start_list = []

    def load_data(self, stock_name, date_predict_start, data_range, slide_range, n_slide):
        all_data = yf.Ticker(stock_name).history(period='max')
        all_data.index = all_data.index.date
        date_predict_start = self._check_start(date_predict_start, all_data)
        predict_start = all_data.index.get_loc(date_predict_start)
        for i in tqdm(range(n_slide)):
            predict_end = predict_start + data_range
            date_predict_end = all_data.iloc[predict_end].name
            data_start = predict_start - data_range
            date_data_start = all_data.iloc[data_start].name
            # print(type(all_data['Close'].iloc[data_start:predict_start].values))
            self.train_data[i, :] = all_data['Close'].iloc[data_start:predict_start].values
            self.test_data[i, :] = all_data['Close'].iloc[predict_start:predict_end].values

            # date_data_start_list.append(date_data_start)
            # date_predict_start_list.append(date_predict_start)
            # date_predict_end_list.append(date_predict_end)
            # test_data_start_list.append(test_data['data_' + str(i)].index[0])
            # data_start = data_start + slide_range
            # predict_start = predict_start + slide_range
            # train_data['data_' + str(i)] = train_data['data_' +
            #                                         str(i)].reset_index(drop=False)
            # test_data['data_' + str(i)] = test_data['data_' +
            #                                         str(i)].reset_index(drop=False)

        # return self.train_data, self.test_data, all_data, self.test_data_start_list
        return self.train_data, self.test_data

    def _check_start(self, date_predict_start, all_data):
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
    slide_range = 5
    n_slide = 5
    # train_data, test_data, all_data, test_data_start_list=\
    #     LoadData.load_data(stock_name, date_predict_start, data_range, slide_range, n_slide)
    # print(train_data)
    # print(test_data)
    # print(all_data)
    # print(test_data_start_list)
    all_data = yf.Ticker(stock_name).history(period='max')
    # all_data.index = all_data.index.date
    train_data = np.ndarray(shape=(n_slide,data_range), dtype=float)
    start = 0
    for i in range(n_slide):
        train_data[i, :] = all_data['Close'].iloc[start:start+data_range].values
        start += slide_range
    print(all_data[0:100])
    # Dataloader = LoadData()
    # train_data, test_data = Dataloader.load_data(stock_name, date_predict_start, data_range, slide_range, n_slide)
    # print(train_data)
    # print(check_start(date_predict_start, all_data))
    # print(type(all_data.index[0]) == datetime.date)