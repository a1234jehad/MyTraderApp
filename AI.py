import cufflinks as cf

from Agents import FreeAgent, TurtleAgent, MovingAverageAgent, SignalRollingAgent, PolicyRadientAgent, QLearningAgent, \
    DoubleRecurrentQLearningAgent, DoubleDuelQlearningAgent, ActorCriticDuelAgent, DuelCuriosityQlearningAgent

cf.go_offline()
import warnings
warnings.simplefilter("ignore")
import yfinance as yf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sns.set()

class AI:
    def __init__(self,Ticker, period, intervals):
        # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        self.stock_df = yf.download(tickers=Ticker, period=period, interval=intervals)
        self.stock_df = self.stock_df[self.stock_df["Volume"] > 0]
        self.Ticker = Ticker
    def evolution_agent(self):
        FreeAgent.free_agent(self.stock_df)

    def turtle_agent(self):
        TurtleAgent.Turtle_Agent(self.stock_df)

    def MA_agent(self):
        MovingAverageAgent.MAA(self.stock_df)

    def signal_rolling_agent(self):
        SignalRollingAgent.SRA(self.stock_df)

    def policy_gradient_agent(self):
        PolicyRadientAgent.PRA(self.stock_df)

    def q_learning_agent(self):
        QLearningAgent.QLA(self.stock_df)

    def double_recurrent_q_lagent(self):
        DoubleRecurrentQLearningAgent.DRQLA(self.stock_df)

    def double_duel_q_learning_agent(self):
        DoubleDuelQlearningAgent.DDQLA(self.stock_df)

    def actor_critic_duel_agent(self):
        ActorCriticDuelAgent.ACDA(self.stock_df)

    def duel_curiosity_q_learning_agent(self):
        DuelCuriosityQlearningAgent.DCQLA(self.stock_df)

    def lstm_5y(self):
        data = self.stock_df
        opn = data[['Open']]
        ds = opn.values
        normalizer = MinMaxScaler(feature_range=(0, 1))
        ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1, 1))
        train_size = int(len(ds_scaled) * 0.70)
        test_size = len(ds_scaled) - train_size
        ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :1]

        def create_ds(dataset, step):
            Xtrain, Ytrain = [], []
            for i in range(len(dataset) - step - 1):
                a = dataset[i:(i + step), 0]
                Xtrain.append(a)
                Ytrain.append(dataset[i + step, 0])
            return np.array(Xtrain), np.array(Ytrain)

        time_stamp = 100
        X_train, y_train = create_ds(ds_train, time_stamp)
        X_test, y_test = create_ds(ds_test, time_stamp)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1, activation='linear'))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)
        loss = model.history.history['loss']
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        train_predict = normalizer.inverse_transform(train_predict)
        test_predict = normalizer.inverse_transform(test_predict)
        test = np.vstack((train_predict, test_predict))
        fut_inp = ds_test[277:]
        fut_inp = fut_inp.reshape(1, -1)
        tmp_inp = list(fut_inp)
        tmp_inp = tmp_inp[0].tolist()
        lst_output = []
        n_steps = 100
        i = 0
        while (i < 30):

            if (len(tmp_inp) > 100):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp = fut_inp.reshape(1, -1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        plot_new = np.arange(1, 101)
        plot_pred = np.arange(101, 131)
        ds_new = ds_scaled.tolist()
        ds_new.extend(lst_output)
        final_graph = normalizer.inverse_transform(ds_new).tolist()
        plt.plot(final_graph, )
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.title("{0} prediction of next month open".format(self.Ticker))
        plt.axhline(y=final_graph[len(final_graph) - 1], color='red', linestyle=':',
                    label='NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph) - 1]), 2)))
        plt.legend()
        plt.show()




ai = AI("MSFT","5y","1d")
# ai.evolution_agent()
# ai.turtle_agent()
# ai.MA_agent()
# ai.signal_rolling_agent()
# ai.policy_gradient_agent()
#ai.q_learning_agent()
# ai.double_recurrent_q_lagent()
#ai.double_duel_q_learning_agent()
#ai.duel_curiosity_q_learning_agent()
ai.lstm_5y()
