import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import math
import keras_tuner as kt
from collections import defaultdict
import keras
def dtw_loss(y_true,y_pred):
    a = tf.map_fn(lambda x: dtw(x[0], x[1]), (y_true, y_pred), dtype=(tf.float32, tf.float32),  fn_output_signature=tf.float32)
    return tf.reduce_mean(a)

def dtw(y_true, y_pred):
    dp = defaultdict(lambda : np.inf)
    # dp = np.ones((y_true.shape[0], y_pred.shape[0]), dtype=np.float32) * np.inf
    # dp = [[float('inf') for i in range(tf.size(y_tr   ue))] for j in range(tf.size(y_pred))]
    x = y_true.shape[0]
    y = y_pred.shape[0]
    dp[0,0] = 0
    for i in range(1,x):
        for j in range(1,y):
            cost = tf.pow((tf.abs(y_true[i]-y_pred[j])),2)
            cost = tf.cast(cost,dtype=np.float32)
            dp[i,j] = tf.reshape(cost + tf.math.minimum(tf.math.minimum(dp[i-1,j], dp[i,j-1]), dp[i-1,j-1]),[])
    return dp[x-1,y-1]


class RNeuralNetwork:
    def __init__(self):
        self.model = models.Sequential([
            layers.LSTM(128, activation="relu"),
            layers.Dense(7, activation='linear', input_shape=(128,))

        ])  
    def build(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss="mean_squared_error")
        return self.model  
    def fit(self,x_train,y_train):
        self.model = self.build()
        # self.model.fit(x_train, y_train, epochs=50, validation_split=0.2)
        self.model.fit(x_train, y_train, epochs=100)
        print(self.model.summary())

    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        self.model.save("model.keras")
    
    def load(self):
        self.model = keras.models.load_model("model.keras")

class OLS:
    def __init__(self):
        self.terms = [lambda x: np.ones(x.shape) , lambda x: x, lambda x: np.square(x), lambda x:np.power(x,3) ]
        self.weights = None
    def fit(self,x_train,y_train):
        X = np.array([i(np.array([j for j in range(len(x_train))])) for i in self.terms]).transpose()
        self.weights = np.linalg.inv(X.transpose() @ X)@X.transpose()@y_train.reshape(-1,1)
    def save(self):
        with open("ols.npy", "wb") as f:
            np.save(f, self.weights, allow_pickle=True)
    def load(self):
        with open("ols.npy", "rb") as f:
            self.weights = np.load(f, allow_pickle=True)
# class RNeuralNetwork:
#     def __init__(self):
#         self.tuner = kt.Hyperband(self.build,
#                     max_epochs=10,
#                     factor=3,
#                     objective="val_loss"
#                    )
#         self.model = None
#     def build(self,hp):
#         hp_units = hp.Int('units', min_value=10, max_value=32, step=2)

#         self.model = models.Sequential([
#             layers.SimpleRNN(hp_units, activation="relu"),
#             layers.Dense(9, activation='linear', input_shape=(hp_units,))

#         ])        
#         hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#         self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=dtw_loss)
#         return self.model

#     def fit(self,x_train,y_train):
#         stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#         self.tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
#         best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
#         self.model = self.tuner.hypermodel.build(best_hps)
#         # self.model.fit(x_train, y_train, epochs=50, validation_split=0.2)
#         self.model.fit(x_train, y_train, epochs=50)
#         print(self.model.summary())

#     def predict(self, X):
#         return self.model.predict(X)

class NeuralNetwork:
    def __init__(self, size):
        self.size = size
        self.tuner = kt.Hyperband(self.build,
                    max_epochs=40,
                    factor=3,
                    objective="val_loss"
                   )
        self.model = None

    def build(self,hp):
        hp_units = hp.Int('units', min_value=1, max_value=32, step=2)
        model = models.Sequential([
        layers.Dense(hp_units, activation='relu', input_shape=(self.size,)),
        layers.Dense(1, activation='linear', input_shape=(10,))
        ])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss="mean_squared_error")
        return model

    #fits data
    def fit(self,x_train,y_train):
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
        # self.tuner.search(x_train, y_train, epochs=50,  callbacks=[stop_early])
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = self.tuner.hypermodel.build(best_hps)
        self.model.fit(x_train, y_train, epochs=50, validation_split=0.2)
        print(self.model.summary())
    #predict from row
    def predict(self, X):
        return self.model.predict(X)
class SlidingWindowTransform:
    def __init__(self, size):
        self.size = size
    def transform(self,data):
        newdata = np.zeros((data.shape[0]-self.size, self.size))
        for i in range(self.size,data.shape[0]):
            newdata[i-self.size,:] = data[i-self.size:i,1]
        return pd.DataFrame(newdata, columns=[f"Data{i}" for i in range(self.size)] )

