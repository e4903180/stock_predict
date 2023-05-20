import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from tqdm import tqdm


class Lstm:
    def __init__(self):
        pass

    def LSTM_model(self, y_length, input_shape):
        model = Sequential()
        model.add(LSTM(units = 256, return_sequences = True, input_shape = input_shape))
        model.add(Dropout(0.4))
        model.add(LSTM(units = 128, return_sequences = True))
        model.add(Dropout(0.4))
        model.add(LSTM(units = 64))
        model.add(Dropout(0.4))
        model.add(Dense(units=y_length))
        return model
    
    def build_Lstm_model(self, x_train, y_length):
        model = {}
        input_shape = (x_train.shape[2],1)
        for i in range(0, x_train.shape[0]):
            model[i] = self.LSTM_model(y_length, input_shape)
            model[i].compile(optimizer='adam', 
                        loss='mean_squared_error')
            # model[i].summary()
        return model
    
    def train_Lstm_model(self, train_data, model, x_train, y_train):
        history = {}
        for i in tqdm(range(0, train_data.shape[0])):
            earlystopper = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 2)
            history[i] = model[i].fit(x_train[i], y_train[i], epochs=25, batch_size=32, 
                                callbacks = [earlystopper], validation_split = 0.2, shuffle=True)
        return history
    
    def predicting_creat_processed_signal_Lstm(self, x_test, test_data, model):
        processed_signal = np.ndarray([])
        for i in test_data:
            predicted_prices = model[i].predict(x_test[i])
            processed_signal[i] = {}
            # processed_signal[i][0] = pd.DataFrame(
            #     {'Close': predicted_prices.flatten()}, index=test_data[i].index[:len(predicted_prices.flatten())])
        return processed_signal