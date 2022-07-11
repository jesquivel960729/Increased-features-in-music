from gc import callbacks
from re import X
import joblib 
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers,Sequential,optimizers
import numpy as np
import matplotlib as plt
import os
from keras.utils import Sequence
from keras.callbacks import History
import keras


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


MUSIC_PATH = "./AB_normalizado.csv"
MUSIC_PATH_TRANS = "yandy_data1.csv"

def load_music_data(music_path = MUSIC_PATH):
    csv_path = os.path.join(music_path)
    return pd.read_csv(csv_path,index_col=0)

AB = load_music_data()
AB = pd.DataFrame(AB)
AB = AB.to_numpy()
dt = load_music_data(MUSIC_PATH_TRANS)
dt = pd.DataFrame(dt)
dt = dt.drop(columns="Artist")

dt = dt.fillna(0)
#dt.to_csv("result.csv")
#print(dt)
H = dt.to_numpy()

AB = AB.flatten()
H = H.flatten()

X_train, X_test, Y_train, Y_test = train_test_split(AB, H, test_size=0.3)
train_gen = DataGenerator(X_train, Y_train, 32)
test_gen = DataGenerator(X_test, Y_test, 32)

model = joblib.load("modelo_entrenado.pkl")

a = model.history.keys()
print(a)

