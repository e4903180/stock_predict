import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from LoadData import LoadData
from Data import LoadData
from tqdm import tqdm

class preprocessing:
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

    def preprocessing_data_Lstm(self, data, x_length, y_length, slide=1):
        shape_1 = len(range(x_length+y_length, data.shape[1], slide))
        x_train = np.ndarray((data.shape[0], shape_1, x_length, 1))
        y_train = np.ndarray((data.shape[0], shape_1, y_length, 1))
        for window in range(0, data.shape[0]):
            scaler = MinMaxScaler(feature_range=(-1,1))
            scaled_data = scaler.fit_transform(data[window].reshape(-1,1))
            num = 0
            for x in range(x_length+y_length, len(scaled_data), slide):
                x_train[window, num] = scaled_data[x-x_length-y_length:x-y_length]
                y_train[window, num] = scaled_data[x-y_length:x]
                num+=1
        return x_train, y_train



if __name__ == '__main__':
    stock_name = "^GSPC"
    date_predict_start = '2020-01-01'
    data_range = 10
    slide_range = 10
    n_slide = 5
    dataloader = LoadData(n_slide, data_range)
    train_data, test_data =\
        dataloader.load_and_split_data(stock_name, date_predict_start, data_range, slide_range, n_slide)
    
    x_length = 2
    y_length = 2
    preprocesser = preprocessing()
    x_train, y_train =\
        preprocesser.preprocessing_data_Lstm(train_data, x_length, y_length, 2)
    print(x_train.shape)
    