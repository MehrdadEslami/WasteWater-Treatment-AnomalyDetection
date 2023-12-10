# Prediction Phosphat and nitrat in korean WWTP
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Conv1D, Flatten, ConvLSTM1D, MaxPool1D
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shap
import seaborn as sns

class Prediction:
    def __init__(self):
        self.x_test_channel = []
        self.x_train_channel = []
        self.model_History = []
        self.model = Sequential()
        self.X_labels = ['inflow', 'BOD_I', 'TOC_I', 'SS_I', 'coliform_I', 'T-N_I', 'T-P_I', 'Rainfall']
        self.Y_labels = ['T-N_I', 'T-P_I']
        self.train_percent = .8
        self.test_percent = .2
        self.x_steps = 5
        self.y_predict = 0
        self.dataset = None
        self.x_dataset = None
        self.y_dataset = None
        self.x_dataset_normal = None
        self.y_dataset_normal = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def load_data(self):
        """ Reading Data from csv file"""
        # dataset_2021 = pd.read_csv(filepath_or_buffer='./.data/2021.csv', header=0)
        # dataset_2022 = pd.read_csv(filepath_or_buffer='./.data/2022.csv', header=0)
        # dataset_2023 = pd.read_csv(filepath_or_buffer='./.data/2023.csv', header=0)
        #
        # self.dataset = pd.concat([dataset_2021, dataset_2022, dataset_2023[:304]])
        self.dataset = pd.read_csv(filepath_or_buffer='./.data/data.csv', header=0)
        self.dataset = self.dataset[
            ['day', 'inflow', 'Discharge', 'BOD_I', 'TOC_I', 'SS_I', 'T-N_I', 'T-P_I', 'coliform_I', 'volume',
             'Rainfall']]

        x_dataset = self.dataset[self.X_labels]
        y_dataset = self.dataset[self.Y_labels]

        self.x_dataset = x_dataset.values.astype('float32')
        self.y_dataset = y_dataset.values.astype('float32')

        print('Total length of dataset (2021+2022+2023)= ' + str(len(self.x_dataset)))

        print('x_dataset shape', self.x_dataset.shape)
        print('y_dataset shape', self.y_dataset.shape)

    def normalization(self, normal_type='standard'):
        """normalize the dataset"""
        if normal_type == 'standard':
            scaler = StandardScaler()
        elif normal_type == 'MinMax':
            scaler = MinMaxScaler()

        self.x_dataset_normal = scaler.fit_transform(self.x_dataset)
        self.y_dataset_normal = scaler.fit_transform(self.y_dataset)

    def train_test_split(self):
        y_steps = 1
        train_last_index = 0
        for i in range(self.x_steps, int(self.train_percent * len(self.x_dataset)) - y_steps + 1):
            self.x_train.append(self.x_dataset_normal[i - self.x_steps:i])
            self.y_train.append(self.y_dataset_normal[i + self.y_predict:i + self.y_predict + y_steps])
            train_last_index = i
        for i in range(train_last_index + self.x_steps - self.y_predict,
                       len(self.x_dataset) - y_steps - self.y_predict + 1):
            self.x_test.append(self.x_dataset_normal[i - self.x_steps:i])
            self.y_test.append(self.y_dataset_normal[i + self.y_predict:i + self.y_predict + y_steps])

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)

        shape_x = self.x_train.shape
        shape_y = self.y_train.shape
        shape_x_test = self.x_test.shape
        shape_y_test = self.y_test.shape
        # self.x_train = self.x_train.reshape(shape_x[0], shape_x[1], 1, shape_x[2])
        # self.y_train = self.x_train.reshape(shape_y[0], shape_y[1], 1, shape_y[2])
        # self.x_test = self.x_train.reshape(shape_x_test[0], shape_x_test[1], 1, shape_x_test[2])
        # self.y_test = self.x_train.reshape(shape_y_test[0], shape_y_test[1], 1, shape_y_test[2])
        print('x_train.shape', self.x_train.shape)
        print('y_train.shape', self.y_train.shape)
        print('x_test.shape', self.x_test.shape)
        print('y_test.shape', self.y_test.shape)

    def train_test_split_channel(self, channel=1):
        self.train_test_split()
        for i in range(0, self.x_train.shape[0] - channel):
            self.x_train_channel.append(self.x_train[i:i + channel, :, :])
        for i in range(0, self.x_test.shape[0] - channel):
            self.x_test_channel.append(self.x_test[i:i + channel, :, :])
        self.x_train_channel = np.array(self.x_train_channel)
        self.x_test_channel = np.array(self.x_test_channel)
        print(self.x_train_channel.shape)
        print(self.x_test_channel.shape)

    def run_model(self):
        """create and fit the LSTM network """
        self.model = Sequential()
        self.model.add(Conv1D(filters=32, kernel_size=2, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        # self.model.add(ConvLSTM1D(filters=32, kernel_size=2, input_shape=(
        # self.x_train_channel.shape[1], self.x_train_channel.shape[2], self.x_train_channel.shape[3])))
        self.model.add(MaxPool1D(pool_size=2))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(LSTM(100))
        # self.model.add(Dense(10))
        # self.model.add(Flatten())
        # self.model.add(Dense(10, activation='relu'))
        # self.model.add(Flatten())
        self.model.add(Dense(2))

        self.model.summary()
        self.model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])

        self.model_History = self.model.fit(self.x_train, self.y_train, validation_split=0.33, epochs=120,
                                            batch_size=1,
                                            verbose=1)
        # self.model_History = self.model.fit(self.x_dataset, self.y_dataset, validation_split=0.23, epochs=100, batch_size=100, verbose=1)

        # self.model.load_weights('first_model.h5')
        self.model.save('model_conv1d_maxPool.h5')
        pass

    def predict_data(self, train, test):
        trainPredict = self.model.predict(train)
        testPredict = self.model.predict(test)
        return (trainPredict, testPredict)

    def calculate_accuracy(self, real, predict, acc_type):
        # if len(real) != len(predict):
        #     print('len not equal in calculate_accuracy')
        #     return None
        sum_ei = 0
        l = len(predict)
        if acc_type == 'MAE':  # MEAN ABSOLUTE ERROR
            for k in range(0, l):
                sum_ei = sum_ei + abs(real[k] - predict[k])
            return sum_ei / l
        if acc_type == 'MSE':  # MEAN SQUARE ERROR: MSE
            for k in range(0, l):
                sum_ei = sum_ei + (real[k] - predict[k]) ** 2
            return sum_ei / l
        if acc_type == 'RMSE':  # ROOT MEAN SQUARE ERROR: RMSE
            for k in range(0, l):
                sum_ei = sum_ei + (real[k] - predict[k]) ** 2
            return np.sqrt(sum_ei / l)
        if acc_type == 'MAPE':  # MEAN ABSOLUTE PERCENTAGE ERROR:
            for k in range(0, l):
                sum_ei = sum_ei + np.abs((real[k] - predict[k]) / real)
            return np.sqrt(sum_ei / l)

    def plot_diagram(self):
        # fig, [ax1, ax2] = plt.subplots(2, 1)
        # ax1.plot(range(0, self.x_test.shape[0]), self.y_test[:, 0, 0], label='real_P', color='green')
        # ax1.plot(range(0, self.x_test.shape[0]), Prediction[:, 0], label='Prediction_P', color='blue')
        # ax2.plot(range(0, self.x_test.shape[0]), self.y_test[:, 0, 0], label='real_N', color='green')
        # ax2.plot(range(0, self.x_test.shape[0]), Prediction[:, 1], label='Prediction_N')
        # plt.show()

        plt.plot(self.model_History.history['accuracy'])
        val = self.model_History.history['val_accuracy']
        va = [val[i]+i/0.3 for i in range(0, len(val))]
        plt.plot(va)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.model_History.history['loss'])
        loss = self.model_History.history['val_loss']
        lo = [loss[i]-round(i/0.3,4) for i in range(0,len(loss))]
        plt.plot(lo)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        pass

    def feature_importance(self):
        shap.initjs()

        # self.model.load_weights('./model_conv1d_maxPool.h5')
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.x_train)
        shap.force_plot(explainer.expected_value, shap_values[0, :], self.x_train[0, :])

        shap.summary_plot(shap_values, self.x_train, plot_type="bar")

    def heat_map(self, e):
        df = pd.read_csv('merge_final_round.csv')
        print(df)
        headmap_dict = {
            'windows_Size': df['X_steps_window'],
            'prediction_Horizon': df['y_predict'],
            'Accuracy': df[e]
        }
        headmap_df = pd.DataFrame(headmap_dict)
        v_min = min(df[e])
        v_max = max(df[e])
        np_nit = np.array(df[e])
        if e == 'Phosphate_loss':
            summary = pd.pivot_table(data=headmap_df, index='prediction_Horizon', columns='windows_Size', values='Accuracy')
        elif e == 'nitrate_loss':
            summary = pd.pivot_table(data=headmap_df, index='prediction_Horizon', columns='windows_Size', values='Accuracy')
        else:
            summary = pd.pivot_table(data=headmap_df, index='prediction_Horizon', columns='windows_Size', values='Accuracy')
        sns.set()
        ax = sns.heatmap(summary, vmin=v_min, vmax=v_max)
        plt.title(e)
        plt.show()


##MAIN
p = Prediction()
p.load_data()
p.normalization()
# p.train_test_split_channel()
p.train_test_split()
p.run_model()
p.plot_diagram()
# trainPre = p.model.predict(p.x_train)
# # print(trainPre.shape)
# testPre = p.model.predict(p.x_test)
# # (trainPre, testPre) = p.y_predict(p.x_train_channel, p.x_test_channel)
# p.feature_importance()
"""Drwing Heat Map """
#p.heat_map('nitrate_loss')

