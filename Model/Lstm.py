from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import numpy as np
from tqdm import tqdm

class Lstm:
    def __init__(self):
        pass

    def lstm(self, x_train, y_train, x_test, y_test, test_data, y_length, epochs):
        """
        Trains and predicts using LSTM model.

        Args:
            x_train (array): Training input data.
            y_train (array): Training target data.
            x_test (array): Test input data.
            y_test (array): Test target data.
            test_data (array): Data for prediction.
            y_length (int): Length of the target data.

        Returns:
            array: Processed signal obtained from predictions.
        """
        model = self._build(x_train, y_length)
        history = self._train(model, x_train, y_train, epochs)
        predicted_y, evaluate = self._predict(x_test, y_test, test_data, model)
        return predicted_y, history, evaluate

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
        model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.4))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=64))
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
        input_shape = (x_train.shape[-2], x_train.shape[-1])
        model = self._construct(y_length, input_shape)
        model.compile(optimizer='adam',
                      loss='mean_squared_error')
        # model[i].summary()
        return model

    def _train(self, model, x_train, y_train, epochs):
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
        earlystopper = EarlyStopping(
            monitor='val_loss', patience=epochs//5, verbose=1)
        history = list()
        for window in range(0, x_train.shape[0]):
            history.append(model.fit(x_train[window], y_train[window], epochs=epochs, batch_size=32,
                                     callbacks=[earlystopper], validation_split=0.2, shuffle=True))
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
        predicted_y = np.ndarray(
            [y_test.shape[0], y_test.shape[1], y_test.shape[2]])
        evaluate = np.ndarray([y_test.shape[0], 2])
        for i in tqdm(range(0, test_data.shape[0])):
            predicted_prices = model.predict(x_test[i])
            evaluate[i] = model.evaluate(x_test[i], y_test[i])
            predicted_y[i] = predicted_prices
        return predicted_y, evaluate
